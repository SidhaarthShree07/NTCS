"""
Tracked Vehicle Calibration - Enhanced LLM-First Approach

This module tracks a SINGLE vehicle from line A to line B, captures screenshots
at both positions WITH visible calibration lines, sends to Gemini Vision to
estimate the real-world distance, then uses that distance to configure homography.

Key improvements over random vehicle sampling:
1. Same vehicle tracked = fair size comparison
2. Screenshots include visible calibration lines = spatial context
3. Road lane markings visible = additional distortion reference
4. LLM estimates distance FIRST, then homography aligns to it
5. Homography becomes a "straightening transform" based on LLM's judgment

Workflow:
  Track vehicle A→B → Capture screenshots → LLM estimates distance → Configure homography
"""

import cv2
import numpy as np
import base64
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

if TYPE_CHECKING:
    from auto_homography import ViewTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def encode_image_to_base64(image: np.ndarray) -> str:
    """Convert OpenCV image to base64 string for Gemini API."""
    success, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        raise ValueError("Failed to encode image to JPEG")
    
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jpg_as_text


def draw_calibration_context(frame: np.ndarray, line_a_y: int, line_b_y: int, 
                              vehicle_bbox: List[float], line_label: str,
                              road_polygon: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Draw calibration lines, vehicle highlight, and trapezoid overlay on frame for LLM context.
    
    Args:
        frame: Original frame
        line_a_y: Y-coordinate of line A (entry/far)
        line_b_y: Y-coordinate of line B (exit/near)
        vehicle_bbox: [x1, y1, x2, y2] of vehicle
        line_label: "LINE A" or "LINE B" to indicate current position
        road_polygon: Optional road polygon to draw cropped trapezoid overlay
    
    Returns:
        Annotated frame with context
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    
    # Draw speed zone trapezoid if road polygon provided
    if road_polygon is not None:
        try:
            from calib_server import create_cropped_speed_polygon
            cropped_trap = create_cropped_speed_polygon(road_polygon, line_a_y, line_b_y)
            # Draw trapezoid with semi-transparent overlay
            overlay = annotated.copy()
            cv2.polylines(overlay, [cropped_trap.astype(np.int32)], True, (0, 255, 255), 3)  # Cyan outline
            cv2.fillPoly(overlay, [cropped_trap.astype(np.int32)], (0, 255, 255))  # Cyan fill
            cv2.addWeighted(overlay, 0.2, annotated, 0.8, 0, annotated)  # Blend for transparency
            
            # Add trapezoid label
            trap_top = int(np.min(cropped_trap[:, 1]))
            trap_center_x = int(np.mean(cropped_trap[:, 0]))
            cv2.putText(annotated, "SPEED ZONE TRAPEZOID", (trap_center_x - 150, trap_top - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        except Exception as e:
            print(f"[TrackedVehicle] Warning: Could not draw trapezoid overlay: {e}")
    
    # Draw both calibration lines (so LLM sees full context)
    cv2.line(annotated, (0, line_a_y), (w, line_a_y), (255, 255, 0), 3)  # Yellow line A
    cv2.line(annotated, (0, line_b_y), (w, line_b_y), (255, 0, 255), 3)  # Magenta line B
    
    # Add line labels
    cv2.putText(annotated, "LINE A (ENTRY/FAR)", (10, line_a_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    cv2.putText(annotated, "LINE B (EXIT/NEAR)", (10, line_b_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)
    
    # Highlight the vehicle with bounding box
    x1, y1, x2, y2 = [int(c) for c in vehicle_bbox]
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green box
    
    # Add current position indicator
    cv2.putText(annotated, f"VEHICLE AT {line_label}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Add pixel distance measurement
    pixel_distance = abs(line_b_y - line_a_y)
    mid_y = (line_a_y + line_b_y) // 2
    cv2.putText(annotated, f"Pixel Distance: {pixel_distance}px", (w - 400, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return annotated


def estimate_distance_from_tracked_vehicle(
    frame_at_line_a: np.ndarray,
    frame_at_line_b: np.ndarray,
    vehicle_bbox_a: List[float],
    vehicle_bbox_b: List[float],
    line_a_y: int,
    line_b_y: int,
    vehicle_type: str,
    gemini_api_key: str,
    road_polygon: Optional[np.ndarray] = None
) -> Dict:
    """
    Estimate real-world distance between lines using LLM vision analysis.
    
    This sends TWO annotated screenshots to Gemini:
    1. Vehicle at line A (with both lines visible)
    2. Vehicle at line B (with both lines visible)
    
    LLM analyzes:
    - Vehicle size change (perspective cue)
    - Road lane markings (distortion reference)
    - Visible calibration lines (spatial context)
    - Camera elevation cues
    
    Args:
        frame_at_line_a: Frame when vehicle crossed line A
        frame_at_line_b: Frame when vehicle crossed line B
        vehicle_bbox_a: [x1, y1, x2, y2] at line A
        vehicle_bbox_b: [x1, y1, x2, y2] at line B
        line_a_y: Y-coordinate of line A
        line_b_y: Y-coordinate of line B
        vehicle_type: Type of vehicle (sedan, truck, bus, motorcycle)
        gemini_api_key: Gemini API key
        road_polygon: Optional road polygon for additional context
    
    Returns:
        Dict with:
        - status: 'success' or 'error'
        - estimated_distance_meters: LLM's distance estimate
        - confidence: LLM's confidence score
        - pixel_distance: Pixel distance between lines
        - vehicle_size_ratio: Apparent size ratio (bbox_b / bbox_a)
        - notes: LLM's reasoning
    """
    logger.info(f"[TrackedCalibration] Estimating distance using tracked {vehicle_type}")
    
    try:
        # Calculate vehicle dimensions in pixels at both positions
        x1_a, y1_a, x2_a, y2_a = vehicle_bbox_a
        x1_b, y1_b, x2_b, y2_b = vehicle_bbox_b
        
        width_a = x2_a - x1_a
        height_a = y2_a - y1_a
        width_b = x2_b - x1_b
        height_b = y2_b - y1_b
        
        # Calculate size ratios (how much bigger vehicle appears at line B vs A)
        width_ratio = width_b / width_a if width_a > 0 else 1.0
        height_ratio = height_b / height_a if height_a > 0 else 1.0
        avg_size_ratio = (width_ratio + height_ratio) / 2
        
        logger.info(f"  Vehicle size ratio: {avg_size_ratio:.2f}x (width: {width_ratio:.2f}, height: {height_ratio:.2f})")
        
        # Pixel distance between lines
        pixel_distance = abs(line_b_y - line_a_y)
        logger.info(f"  Pixel distance: {pixel_distance}px")
        
        # Annotate frames with context (including trapezoid overlay if road polygon provided)
        annotated_a = draw_calibration_context(frame_at_line_a, line_a_y, line_b_y, 
                                                 vehicle_bbox_a, "LINE A", road_polygon)
        annotated_b = draw_calibration_context(frame_at_line_b, line_a_y, line_b_y,
                                                 vehicle_bbox_b, "LINE B", road_polygon)

        # Save annotated images to disk for manual inspection
        out_dir = os.path.join(os.path.dirname(__file__), 'calib')
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        a_path = os.path.join(out_dir, f"tracked_vehicle_line_a_{ts}.jpg")
        b_path = os.path.join(out_dir, f"tracked_vehicle_line_b_{ts}.jpg")
        try:
            cv2.imwrite(a_path, annotated_a)
            cv2.imwrite(b_path, annotated_b)
            logger.info(f"  Saved annotated images: A={a_path}, B={b_path}")
        except Exception as save_err:
            logger.warning(f"  Failed to save annotated images: {save_err}")
        
        # Encode images to base64
        img_a_b64 = encode_image_to_base64(annotated_a)
        img_b_b64 = encode_image_to_base64(annotated_b)
        
        logger.info(f"  Encoded images: A={len(img_a_b64)} chars, B={len(img_b_b64)} chars")
        
        # Get image dimensions for context
        img_height, img_width = frame_at_line_a.shape[:2]
        
        # Build comprehensive prompt for Gemini
        prompt = f"""You are a computer vision expert analyzing traffic camera calibration for DISTANCE MEASUREMENT.

═══════════════════════════════════════════════════════════════════════
TASK: Calculate REAL-WORLD DISTANCE between two calibration lines
═══════════════════════════════════════════════════════════════════════

📸 IMAGES PROVIDED:
• Image 1: Vehicle at LINE A (yellow line, Y={line_a_y})
• Image 2: SAME vehicle at LINE B (magenta line, Y={line_b_y})
• Both lines are VISIBLE in both images for spatial reference

🎯 CRITICAL - VEHICLE IDENTIFICATION:
• The vehicle of interest is outlined by a **GREEN BOUNDING BOX** in BOTH images
• This vehicle is labeled "VEHICLE AT LINE A" in Image 1 and "VEHICLE AT LINE B" in Image 2
• There may be OTHER vehicles visible in the images - **IGNORE THEM COMPLETELY**
• Use ONLY the vehicle inside the GREEN BOX for all measurements, identification, and calculations
• Do NOT consider any other vehicles, partial vehicles, reflections, shadows, or objects
• The green box marks the SAME TRACKED VEHICLE at two different positions

📊 MEASURED PIXEL DATA:
• Line A to Line B: {pixel_distance} pixels vertical distance
• Image resolution: {img_width} × {img_height} pixels
• Vehicle pixel size at LINE A: {width_a:.0f}px wide × {height_a:.0f}px tall
• Vehicle pixel size at LINE B: {width_b:.0f}px wide × {height_b:.0f}px tall
• Apparent size ratio: {avg_size_ratio:.2f}× (near/far)

═══════════════════════════════════════════════════════════════════════
STEP 1: IDENTIFY VEHICLE MODEL & REAL-WORLD DIMENSIONS
═══════════════════════════════════════════════════════════════════════

⚠️ REMINDER: Only analyze the vehicle inside the GREEN BOUNDING BOX!

Carefully examine the GREEN-BOXED vehicle in BOTH images:

1. **Determine vehicle category**:
   - Sedan (Honda Civic, Toyota Camry, etc.)
   - SUV/Crossover (Honda CR-V, Toyota RAV4, etc.)
   - Truck/Pickup (Ford F-150, Toyota Tacoma, etc.)
   - Van/Minivan (Honda Odyssey, Toyota Sienna, etc.)
   - Bus (city bus, coach, etc.)
   - Motorcycle/Scooter

2. **Identify specific make/model if possible**:
   - Look at body shape, grill design, proportions
   - Note any visible badges or distinctive features

3. **Determine REAL-WORLD dimensions**:
   Use standard dimensions for identified vehicle type:
   
   **SEDANS** (most common):
   • Length: 4.5-4.9m (avg: 4.7m)
   • Width: 1.7-1.9m (avg: 1.8m)  
   • Height: 1.4-1.5m (avg: 1.45m)
   Examples: Honda Civic (4.63m), Toyota Camry (4.88m), Hyundai Elantra (4.64m)
   
   **SUVS/CROSSOVERS**:
   • Length: 4.6-5.0m (avg: 4.8m)
   • Width: 1.8-2.0m (avg: 1.9m)
   • Height: 1.6-1.8m (avg: 1.7m)
   Examples: Honda CR-V (4.69m), Toyota RAV4 (4.60m), Mazda CX-5 (4.55m)
   
   **TRUCKS/PICKUPS**:
   • Length: 5.3-5.9m (avg: 5.6m)
   • Width: 2.0-2.2m (avg: 2.1m)
   • Height: 1.8-2.0m (avg: 1.9m)
   Examples: Ford F-150 (5.89m), Toyota Tacoma (5.39m), Chevrolet Silverado (5.84m)
   
   **BUSES**:
   • Length: 10-12m (avg: 11m)
   • Width: 2.5m
   • Height: 3.0-3.5m
   
   **MOTORCYCLES**:
   • Length: 2.0-2.3m (avg: 2.1m)
   • Width: 0.7-0.9m (avg: 0.8m)
   • Height: 1.1-1.3m (avg: 1.2m)

═══════════════════════════════════════════════════════════════════════
STEP 2: DETERMINE CAMERA ANGLE & POSITION
═══════════════════════════════════════════════════════════════════════

Analyze camera perspective from the images:

1. **Camera Height Estimation**:
   - Look at horizon line position (if visible)
   - Examine how much of vehicle TOP is visible
   - More top visible = higher camera elevation
   
   Camera height categories:
   • **LOW** (2-5m): Street level, slight top visible
   • **MEDIUM** (5-15m): Traffic pole height, moderate top view
   • **HIGH** (15-25m): Highway gantry, significant top view
   • **VERY HIGH** (25m+): Building/tower mount, bird's eye view

2. **Camera Angle**:
   - **Straight ahead**: Traffic flows toward/away from camera (TYPICAL)
   - **Oblique angle**: Traffic at 30-60° angle to camera axis
   - **Near-perpendicular**: Side view of traffic (RARE)
   
   For this system: Assume traffic flows ALONG camera axis (toward or away)

3. **Viewing Direction**:
   - LINE A is FAR (entry) → Vehicle appears SMALLER
   - LINE B is NEAR (exit) → Vehicle appears LARGER
   - Size ratio indicates perspective compression

═══════════════════════════════════════════════════════════════════════
STEP 3: CALCULATE DISTANCE USING PERSPECTIVE GEOMETRY
═══════════════════════════════════════════════════════════════════════

⚠️ REMINDER: Measure pixel dimensions from the GREEN-BOXED vehicle only!

Use the following methodology:

**METHOD A: Size Ratio Analysis** (Primary)

For cameras where traffic flows along viewing axis:

1. Use vehicle's REAL dimension (from Step 1 - identified from GREEN BOX)
2. Calculate pixel-to-meter ratio at each line:
   
   At LINE A (far):
   pixels_per_meter_A = vehicle_pixels_A / vehicle_real_dimension
   
   At LINE B (near):
   pixels_per_meter_B = vehicle_pixels_B / vehicle_real_dimension

3. Use perspective formula - THIS IS THE CORE CALCULATION:
   
   For elevated camera with traffic along axis:
   
   Step 3a: Calculate meters per pixel at each position
   meters_per_pixel_A = 1 / pixels_per_meter_A
   meters_per_pixel_B = 1 / pixels_per_meter_B
   
   Step 3b: Calculate scale difference
   scale_difference = meters_per_pixel_A - meters_per_pixel_B
   
   Step 3c: Apply camera height adjustment factor
   Adjustment factor based on detected camera height:
   • Low (2-5m): factor = 0.8-1.0
   • Medium (5-15m): factor = 1.0-1.3  
   • High (15-25m): factor = 1.3-1.8
   • Very High (25m+): factor = 1.8-2.5
   
   Step 3d: Final distance calculation
   distance_meters = (1 / scale_difference) × adjustment_factor

4. **CRITICAL - NO HEURISTICS**:
   Calculate distance using ONLY the geometric formula above (steps 3a-3d).
   Do NOT use pre-defined size ratio → distance lookup tables.
   Do NOT say "2.5x ratio means 150-200m" - that's heuristic guessing.
   ALWAYS compute using actual pixel measurements and perspective geometry.
   
   The size ratio is a RESULT of perspective, not a shortcut to estimate distance.
   
   **DETAILED EXAMPLE CALCULATION WITH FORMULA BREAKDOWN**:
   
   Given:
   - Vehicle identified: Honda Civic, real width = 1.80m (from STEP 1)
   - At Line A (far): GREEN BOX shows vehicle is 82 pixels wide
   - At Line B (near): GREEN BOX shows vehicle is 175 pixels wide
   - Camera detected: HIGH elevation (approximately 18m above ground)
   
   Step-by-step calculation:
   
   1. Calculate pixels per meter at Line A:
      PPM_A = vehicle_pixels_A / vehicle_real_width
      PPM_A = 82 px / 1.80 m
      PPM_A = 45.6 px/m
   
   2. Calculate pixels per meter at Line B:
      PPM_B = vehicle_pixels_B / vehicle_real_width
      PPM_B = 175 px / 1.80 m
      PPM_B = 97.2 px/m
   
   3. Convert to meters per pixel (reciprocal):
      MPP_A = 1 / PPM_A = 1 / 45.6 = 0.0219 m/px (far position)
      MPP_B = 1 / PPM_B = 1 / 97.2 = 0.0103 m/px (near position)
   
   4. Calculate scale difference (perspective gradient):
      scale_difference = MPP_A - MPP_B
      scale_difference = 0.0219 - 0.0103
      scale_difference = 0.0116 m/px
   
   5. Apply camera height adjustment factor:
      Camera category: HIGH (15-25m elevation)
      adjustment_factor = 1.5 (from the table above)
   
   6. Calculate final distance using perspective formula:
      distance = (1 / scale_difference) × adjustment_factor
      distance = (1 / 0.0116) × 1.5
      distance = 86.2 × 1.5
      distance = 129.3 meters
   
   ✅ Final answer: The distance between Line A and Line B is 129.3 meters
   
   This calculation uses pure geometry - the vehicle's known real-world size
   (1.80m) acts as a "measuring stick" at two different distances, and the
   perspective compression tells us how far apart those positions are.

**METHOD B: Road Marking Verification** (Secondary)

1. Count visible lane dashes between lines (if visible)
2. Standard highway dashes: 3m dash + 9m gap = 12m per cycle
3. Count cycles × 12m = approximate distance
4. Use as sanity check for Method A

**METHOD C: Horizon/Vanishing Point** (Tertiary)

1. If horizon visible, measure its Y-position
2. Higher horizon = higher camera
3. Use to refine camera height estimate

═══════════════════════════════════════════════════════════════════════
STEP 4: VERIFY & ADJUST (NO PRIORS; NO TYPICAL-RANGE ASSUMPTIONS)
═══════════════════════════════════════════════════════════════════════

Perform consistency checks:

1. **No prior ranges**:
    - Do not apply any “typical” distance ranges or speed assumptions.
    - This system must handle arbitrary scenes (urban, highway, long-range, short-range).
    - Report the computed result even if it’s unusually large or small; if unusual, reflect that only in the confidence and reasoning.

2. **Cross-Method Validation**:
    - Do geometric size analysis and road-marking count (if visible) agree?
    - If there is major discrepancy, lower confidence and clearly explain why.

3. **Zoom/Field of View (FOV) Note**:
    - Identify FOV category (wide/normal/telephoto) for context.
    - Do NOT modify the geometric result using FOV; at most, adjust confidence.

═══════════════════════════════════════════════════════════════════════
IMPORTANT: UNIVERSAL CAMERA POSITION HANDLING
═══════════════════════════════════════════════════════════════════════

This system works for ANY forward/backward traffic view:
✓ Traffic coming TOWARD camera (LINE A far, LINE B near)
✓ Traffic going AWAY from camera (LINE A entrance, LINE B exit)
✓ Any camera height (2m to 50m+)
✓ Any camera angle (straight or oblique up to 60°)

NOT designed for:
✗ Side-view cameras (traffic perpendicular to camera axis)
✗ Overhead bird's-eye (no perspective compression)

═══════════════════════════════════════════════════════════════════════
OUTPUT REQUIREMENTS
═══════════════════════════════════════════════════════════════════════

⚠️ CRITICAL REMINDER: All measurements and identification must be based ONLY on the vehicle inside the GREEN BOUNDING BOX in both images. Ignore all other vehicles completely.

Return ONLY valid JSON (no markdown, no extra text):

{{
  "vehicle_identification": {{
    "category": "<sedan|suv|truck|bus|motorcycle>",
    "specific_model": "<make/model if identifiable, or 'unknown'>",
    "real_length_meters": <float>,
    "real_width_meters": <float>,
    "real_height_meters": <float>,
    "confidence_in_id": <0.0-1.0>,
    "note": "Measured from GREEN-BOXED vehicle only"
  }},
  "camera_analysis": {{
    "estimated_height_meters": <float>,
    "height_category": "<low|medium|high|very_high>",
    "viewing_angle": "<straight|oblique>",
    "field_of_view": "<wide|normal|telephoto>",
    "perspective_compression": "<weak|moderate|strong|extreme>"
  }},
  "distance_calculation": {{
    "method_a_geometric_meters": <float>,
    "method_b_road_markings_meters": <float or null if not visible>,
    "methods_agree": <true|false>,
    "final_distance_meters": <float>,
    "calculation_details": "<brief explanation of math used>"
  }},
  "confidence": <float 0.0-1.0>,
  "reasoning": "<CONCISE explanation (2-3 sentences max): vehicle ID, camera position, and final distance>",
  "warnings": "<any concerns or 'none'>"
}}

IMPORTANT: Keep "reasoning" field BRIEF (under 200 characters). No repeated calculations or verbose explanations.

CRITICAL: Use actual perspective geometry, not just heuristic ratios!

═══════════════════════════════════════════════════════════════════════
CALCULATION METHODOLOGY ENFORCEMENT
═══════════════════════════════════════════════════════════════════════

You MUST follow this exact calculation process:

1. ✅ CORRECT: "Vehicle width is 1.80m. At line A it's 82px (45.6 px/m), at line B it's 175px (97.2 px/m). Scale difference: 0.0116 m/px. Camera: HIGH (factor=1.5). Distance = 129.3m"

2. ❌ WRONG: "Size ratio is 2.13x which typically means 100-150m for elevated cameras"

3. ❌ WRONG: "Based on the size ratio guidelines, this looks like 200-300m"

4. ❌ WRONG: "The 4x size ratio suggests approximately 300m+ depth"

The size ratio is an OBSERVATION, not a CALCULATION METHOD.
Always compute using: distance = (1/scale_difference) × camera_adjustment_factor

═══════════════════════════════════════════════════════════════════════"""

        # Log the exact prompt that will be sent to LLM for reproducibility
        logger.info("[TrackedCalibration] Prompt to Gemini (distance):\n" + prompt)
        logger.info(f"[TrackedCalibration] Sending images to Gemini:\n  Image A: {a_path}\n  Image B: {b_path}")

        # Initialize Gemini
        logger.info("[TrackedCalibration] Calling Gemini Vision API...")
        chat = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=gemini_api_key,
            temperature=0.1,  # Very low for consistent geometric calculations
            max_output_tokens=4096,  # Increased to handle longer responses
        )
        
        # Send both images with prompt
        human_message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_a_b64}"}
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b_b64}"}
                }
            ]
        )
        
        response = chat.invoke([human_message])
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        logger.info(f"[TrackedCalibration] Gemini response: {response_text}")
        
        # Parse JSON response
        cleaned_text = response_text.strip()
        if cleaned_text.startswith("```json"):
            lines = cleaned_text.splitlines()
            cleaned_text = "\n".join(lines[1:])
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
        
        cleaned_text = cleaned_text.strip()
        
        # Extract JSON - handle truncated responses robustly
        if '{' in cleaned_text and '}' in cleaned_text:
            json_start = cleaned_text.find('{')
            json_end = cleaned_text.rfind('}')
            json_str = cleaned_text[json_start:json_end+1]
            
            # Try to parse; if it fails due to truncation, attempt to repair
            try:
                result = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(f"[TrackedCalibration] Initial JSON parse failed: {e}")
                logger.info("[TrackedCalibration] Attempting to repair truncated JSON...")
                
                # Common issue: truncated string in reasoning/warnings field
                # Try to close any open strings and objects
                repaired = json_str
                
                # Count braces to see if we need to close
                open_braces = repaired.count('{') - repaired.count('}')
                open_brackets = repaired.count('[') - repaired.count(']')
                
                # If string is not closed, try to close it
                if repaired.count('"') % 2 != 0:
                    repaired += '"'
                
                # Close any open brackets/braces
                repaired += ']' * open_brackets
                repaired += '}' * open_braces
                
                try:
                    result = json.loads(repaired)
                    logger.info("[TrackedCalibration] Successfully repaired truncated JSON")
                except json.JSONDecodeError as e2:
                    logger.error(f"[TrackedCalibration] JSON repair failed: {e2}")
                    raise ValueError(f"Could not parse JSON response (original error: {e}, repair error: {e2})")
        else:
            raise ValueError(f"No JSON found in response: {cleaned_text}")
        
        # Parse structured response
        vehicle_id = result.get('vehicle_identification', {})
        camera_analysis = result.get('camera_analysis', {})
        distance_calc = result.get('distance_calculation', {})
        
        # Extract final distance (try multiple paths for compatibility)
        estimated_distance = None
        if distance_calc and 'final_distance_meters' in distance_calc:
            estimated_distance = float(distance_calc['final_distance_meters'])
        elif 'estimated_distance_meters' in result:
            estimated_distance = float(result['estimated_distance_meters'])
        else:
            raise ValueError("No distance estimate found in response")
        
        confidence = float(result.get('confidence', 0.5))
        
        if estimated_distance <= 0:
            raise ValueError(f"Invalid distance estimate: {estimated_distance}")
        
        # Log detailed results
        logger.info(f"[TrackedCalibration] ✅ LLM Analysis Complete:")
        logger.info(f"  📏 Distance: {estimated_distance:.2f}m (confidence: {confidence:.2f})")
        
        if vehicle_id:
            logger.info(f"  🚗 Vehicle: {vehicle_id.get('category', 'unknown')} - {vehicle_id.get('specific_model', 'unknown')}")
            logger.info(f"      Real dimensions: {vehicle_id.get('real_length_meters', 0):.2f}m × {vehicle_id.get('real_width_meters', 0):.2f}m")
        
        if camera_analysis:
            logger.info(f"  📸 Camera: {camera_analysis.get('height_category', 'unknown')} ({camera_analysis.get('estimated_height_meters', 0):.1f}m)")
            logger.info(f"      Perspective: {camera_analysis.get('perspective_compression', 'unknown')}")
        
        if distance_calc:
            logger.info(f"  🧮 Calculation: {distance_calc.get('calculation_details', 'N/A')}")
            if distance_calc.get('method_b_road_markings_meters'):
                logger.info(f"      Road markings verify: {distance_calc['method_b_road_markings_meters']:.1f}m")
        
        logger.info(f"  💡 Reasoning: {result.get('reasoning', 'N/A')}")
        
        if result.get('warnings') and result['warnings'] != 'none':
            logger.warning(f"  ⚠️  Warnings: {result['warnings']}")
        
        return {
            'status': 'success',
            'estimated_distance_meters': estimated_distance,
            'confidence': confidence,
            'pixel_distance': pixel_distance,
            'vehicle_size_ratio': avg_size_ratio,
            'vehicle_identification': vehicle_id,
            'camera_analysis': camera_analysis,
            'distance_calculation': distance_calc,
            'reasoning': result.get('reasoning', ''),
            'warnings': result.get('warnings', 'none'),
            'perspective_assessment': camera_analysis.get('perspective_compression', 'unknown'),
            'road_marking_observations': distance_calc.get('method_b_road_markings_meters', None),
            'annotated_frame_a': annotated_a,
            'annotated_frame_b': annotated_b,
            'annotated_frame_a_path': a_path,
            'annotated_frame_b_path': b_path
        }
        
    except Exception as e:
        logger.error(f"[TrackedCalibration] Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'status': 'error',
            'message': str(e),
            'estimated_distance_meters': None
        }


def configure_homography_from_llm_distance(
    road_polygon: np.ndarray,
    llm_distance_meters: float,
    line_a_y: int,
    line_b_y: int,
    img_width: int,
    img_height: int,
    road_width_estimate_meters: float = 10.0
) -> Optional['ViewTransformer']:
    """
    Configure homography transformation using LLM's distance estimate.
    
    This creates a perspective transformation that "straightens" the road
    based on the LLM's judgment of real-world distance.
    
    Args:
        road_polygon: Road polygon points
        llm_distance_meters: Distance estimated by LLM
        line_a_y: Y-coordinate of line A
        line_b_y: Y-coordinate of line B
        img_width: Image width
        img_height: Image height
        road_width_estimate_meters: Estimated road width (default 10m for highway)
    
    Returns:
        ViewTransformer object configured with LLM-aligned homography
    """
    logger.info("[TrackedCalibration] Configuring homography from LLM distance...")
    
    try:
        from auto_homography import ViewTransformer
        
        # Extract road polygon bounds
        polygon_top_y = int(np.min(road_polygon[:, 1]))
        polygon_bottom_y = int(np.max(road_polygon[:, 1]))
        polygon_left_x = int(np.min(road_polygon[:, 0]))
        polygon_right_x = int(np.max(road_polygon[:, 0]))
        
        logger.info(f"  Road polygon: X={polygon_left_x}-{polygon_right_x}, Y={polygon_top_y}-{polygon_bottom_y}")
        
        # Calculate the proportion of road covered by calibration lines
        line_a_offset_from_top = line_a_y - polygon_top_y
        line_b_offset_from_top = line_b_y - polygon_top_y
        total_polygon_height = polygon_bottom_y - polygon_top_y
        
        # LLM measured distance between lines
        # Extrapolate to full road depth
        line_section_height = line_b_y - line_a_y
        
        if line_section_height <= 0:
            raise ValueError("Invalid line positions")
        
        # Estimate full road depth (extrapolate from LLM measurement)
        # Proportion of road depth covered by lines
        line_section_proportion = line_section_height / total_polygon_height
        
        # If lines cover X% of image, and LLM says that's Y meters,
        # then full road depth = Y / X
        estimated_full_depth = llm_distance_meters / line_section_proportion
        
        logger.info(f"  LLM distance between lines: {llm_distance_meters:.2f}m")
        logger.info(f"  Lines cover {line_section_proportion*100:.1f}% of polygon height")
        logger.info(f"  Extrapolated full road depth: {estimated_full_depth:.2f}m")
        
        # Define SOURCE region (image coordinates - use full polygon)
        SOURCE = np.array([
            [polygon_left_x, polygon_top_y],      # Top-left
            [polygon_right_x, polygon_top_y],     # Top-right
            [polygon_right_x, polygon_bottom_y],  # Bottom-right
            [polygon_left_x, polygon_bottom_y]    # Bottom-left
        ], dtype=np.float32)
        
        # Define TARGET region (real-world meters)
        TARGET = np.array([
            [0, 0],                                    # Top-left (origin)
            [road_width_estimate_meters, 0],          # Top-right
            [road_width_estimate_meters, estimated_full_depth],  # Bottom-right
            [0, estimated_full_depth]                 # Bottom-left
        ], dtype=np.float32)
        
        logger.info(f"  SOURCE region: {SOURCE.tolist()}")
        logger.info(f"  TARGET region: {TARGET.tolist()}")
        
        # Create ViewTransformer
        transformer = ViewTransformer(source=SOURCE, target=TARGET)
        
        logger.info("[TrackedCalibration] ✅ Homography configured successfully!")
        logger.info(f"  Transformation matrix: {transformer.m.tolist()}")
        
        return transformer
        
    except Exception as e:
        logger.error(f"[TrackedCalibration] Error configuring homography: {e}")
        import traceback
        traceback.print_exc()
        return None
