# feature_extractor_class.py
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from math import degrees, atan2, sqrt
import os
import collections

# --- dlib 关键点索引 (基于 68 点模型) ---
LandmarkIndices = collections.namedtuple('LandmarkIndices', [
    'JAW', 'RIGHT_EYEBROW', 'LEFT_EYEBROW', 'NOSE_BRIDGE', 'NOSE_TIP',
    'RIGHT_EYE', 'LEFT_EYE', 'MOUTH_OUTER', 'MOUTH_INNER'
])

indices = LandmarkIndices(
    JAW=list(range(0, 17)),
    RIGHT_EYEBROW=list(range(17, 22)),
    LEFT_EYEBROW=list(range(22, 27)),
    NOSE_BRIDGE=list(range(27, 31)),
    NOSE_TIP=list(range(31, 36)),
    RIGHT_EYE=list(range(36, 42)),
    LEFT_EYE=list(range(42, 48)),
    MOUTH_OUTER=list(range(48, 60)),
    MOUTH_INNER=list(range(60, 68))
)


class FacialFeatureExtractor:
    def __init__(self, face_detector_path, landmark_predictor_path):
        """
        Initializes the FacialFeatureExtractor with paths to dlib models.
        Args:
            face_detector_path (str): Path to the dlib face detector model (e.g., mmod_human_face_detector.dat).
                                      Set to "" or None to use HOG detector.
            landmark_predictor_path (str): Path to the dlib landmark predictor model (shape_predictor_68_face_landmarks.dat).
        """
        self.detector = None
        self.predictor = None
        self.models_loaded = False
        self.indices = indices  # Make indices accessible within the class

        print("Initializing FacialFeatureExtractor...")
        if not os.path.isfile(landmark_predictor_path):
            print(f"FATAL ERROR: Landmark predictor model not found at '{landmark_predictor_path}'")
            # Optionally raise FileNotFoundError to prevent class instantiation with missing models
            # raise FileNotFoundError(f"Landmark predictor model not found at '{landmark_predictor_path}'")
            return

        try:
            self.predictor = dlib.shape_predictor(landmark_predictor_path)
            print("Landmark predictor loaded.")
        except Exception as e:
            print(f"FATAL ERROR: Could not load Landmark predictor model from '{landmark_predictor_path}': {e}")
            return

        use_hog = not (
                    face_detector_path and os.path.isfile(face_detector_path) and face_detector_path.lower().endswith(
                ".dat"))
        if not use_hog and not os.path.isfile(face_detector_path):
            print(
                f"Warning: Face detector model specified ('{face_detector_path}') but not found. Falling back to HOG.")
            use_hog = True

        try:
            if use_hog:
                self.detector = dlib.get_frontal_face_detector()
                print("Using default HOG face detector.")
            else:
                print(f"Attempting to load CNN face detector (mmod) from: {face_detector_path}")
                self.detector = dlib.cnn_face_detection_model_v1(face_detector_path)
                print("Using CNN face detector (mmod).")
            self.models_loaded = True  # Models are considered loaded if detector and predictor are set
        except Exception as e:
            print(f"ERROR loading dlib face detector: {e}")
            if not use_hog and self.detector is None:  # CNN was intended but failed
                print("Attempting to fall back to HOG detector...")
                try:
                    self.detector = dlib.get_frontal_face_detector()
                    print("Successfully loaded HOG detector as fallback.")
                    self.models_loaded = True
                except Exception as hog_e:
                    print(f"ERROR loading HOG detector as fallback: {hog_e}")
                    self.models_loaded = False  # Fallback also failed
            else:  # HOG was default but failed, or predictor failed earlier
                self.models_loaded = False

        # Final check if both models are loaded
        if self.predictor is None or self.detector is None:
            self.models_loaded = False

        if not self.models_loaded:
            print("CRITICAL: dlib models could not be fully loaded in FacialFeatureExtractor.")
        else:
            print("FacialFeatureExtractor initialized and models loaded successfully.")

    @staticmethod
    def landmarks_to_np(shape, dtype="int"):
        coords = np.zeros((shape.num_parts, 2), dtype=dtype)
        for i in range(0, shape.num_parts):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    @staticmethod
    def get_distance(p1, p2):
        return dist.euclidean(p1, p2)

    @staticmethod
    def get_midpoint(p1, p2):
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

    @staticmethod
    def get_angle_degrees(p1, p2):
        x_diff = p2[0] - p1[0]
        y_diff = p2[1] - p1[1]
        return degrees(atan2(-y_diff, x_diff))

    def analyze_face_proportions(self, landmarks):
        features = {}
        if not landmarks.size or landmarks.shape[0] < 17: return features  # Not enough points for JAW
        try:
            face_width = self.get_distance(landmarks[self.indices.JAW[0]], landmarks[self.indices.JAW[16]])
            features['face_width'] = face_width

            # Ensure NOSE_BRIDGE is accessible
            if landmarks.shape[0] > max(self.indices.JAW[8], self.indices.NOSE_BRIDGE[0]):
                face_height_jaw_nose = self.get_distance(landmarks[self.indices.JAW[8]],
                                                         landmarks[self.indices.NOSE_BRIDGE[0]])
                features['face_height_jaw_nose'] = face_height_jaw_nose
                if face_width > 0:
                    features['face_aspect_ratio_v1'] = face_height_jaw_nose / face_width
                else:
                    features['face_aspect_ratio_v1'] = 0.0
            else:
                features['face_height_jaw_nose'] = 0.0
                features['face_aspect_ratio_v1'] = 0.0

            if landmarks.shape[0] > max(self.indices.RIGHT_EYE[0], self.indices.LEFT_EYE[3], self.indices.JAW[8]):
                eye_outer_mid = self.get_midpoint(landmarks[self.indices.RIGHT_EYE[0]],
                                                  landmarks[self.indices.LEFT_EYE[3]])
                face_height_jaw_eyes = self.get_distance(landmarks[self.indices.JAW[8]], eye_outer_mid)
                features['face_height_jaw_eyes'] = face_height_jaw_eyes
                if face_width > 0:
                    features['face_aspect_ratio_v2'] = face_height_jaw_eyes / face_width
                else:
                    features['face_aspect_ratio_v2'] = 0.0
            else:
                features['face_height_jaw_eyes'], features['face_aspect_ratio_v2'] = 0.0, 0.0

            if landmarks.shape[0] > max(self.indices.JAW[4], self.indices.JAW[8], self.indices.JAW[12]):
                angle_left = self.get_angle_degrees(landmarks[self.indices.JAW[4]], landmarks[self.indices.JAW[8]])
                angle_right = self.get_angle_degrees(landmarks[self.indices.JAW[12]], landmarks[self.indices.JAW[8]])
                jaw_angle = abs(angle_left - angle_right)  # Can be > 180, adjust if needed
                features['jaw_angle'] = jaw_angle
            else:
                features['jaw_angle'] = 0.0

            if landmarks.shape[0] > max(self.indices.JAW[1], self.indices.JAW[15]):
                cheek_width = self.get_distance(landmarks[self.indices.JAW[1]], landmarks[self.indices.JAW[15]])
                features['cheek_width_ratio'] = cheek_width / face_width if face_width > 0 else 0.0
            else:
                features['cheek_width_ratio'] = 0.0
        except IndexError as e:
            print(f"Error: Landmark index out of bounds in face proportions: {e}")
        except Exception as e:
            print(f"Error in face proportions: {e}")
        return features

    @staticmethod
    def calculate_eye_aspect_ratio(eye_landmarks):
        if eye_landmarks is None or len(eye_landmarks) != 6: return 0.0
        A = FacialFeatureExtractor.get_distance(eye_landmarks[1], eye_landmarks[5])
        B = FacialFeatureExtractor.get_distance(eye_landmarks[2], eye_landmarks[4])
        C = FacialFeatureExtractor.get_distance(eye_landmarks[0], eye_landmarks[3])
        if C == 0: return 0.0
        return (A + B) / (2.0 * C)

    def analyze_eyes(self, landmarks, face_width):
        features = {}
        if landmarks.shape[0] < max(self.indices.LEFT_EYE[-1], self.indices.RIGHT_EYE[-1]) + 1: return features
        try:
            left_eye = landmarks[self.indices.LEFT_EYE]
            right_eye = landmarks[self.indices.RIGHT_EYE]

            left_ear = self.calculate_eye_aspect_ratio(left_eye)
            right_ear = self.calculate_eye_aspect_ratio(right_eye)
            features['eye_aspect_ratio_left'], features['eye_aspect_ratio_right'] = left_ear, right_ear
            features['eye_aspect_ratio_avg'] = (left_ear + right_ear) / 2.0

            left_eye_center = self.get_midpoint(left_eye[0], left_eye[3])
            right_eye_center = self.get_midpoint(right_eye[0], right_eye[3])
            ipd = self.get_distance(left_eye_center, right_eye_center)
            features['interpupillary_distance'] = ipd
            features['eye_spacing_ratio'] = ipd / face_width if face_width > 0 else 0.0

            left_eye_angle = self.get_angle_degrees(left_eye[3], left_eye[0])  # (end, start)
            right_eye_angle = self.get_angle_degrees(right_eye[0], right_eye[3])  # (start, end)
            features['eye_slant_angle_left'], features['eye_slant_angle_right'] = left_eye_angle, right_eye_angle
            features['eye_slant_avg'] = (left_eye_angle + right_eye_angle) / 2.0
        except IndexError as e:
            print(f"Error: Landmark index out of bounds in eye analysis: {e}")
        except Exception as e:
            print(f"Error in eye analysis: {e}")
        return features

    def analyze_eyebrows(self, landmarks):
        features = {}
        if landmarks.shape[0] < max(self.indices.LEFT_EYEBROW[-1], self.indices.RIGHT_EYEBROW[-1],
                                    self.indices.LEFT_EYE[-1], self.indices.RIGHT_EYE[-1]) + 1: return features
        try:
            left_brow, right_brow = landmarks[self.indices.LEFT_EYEBROW], landmarks[self.indices.RIGHT_EYEBROW]
            left_eye, right_eye = landmarks[self.indices.LEFT_EYE], landmarks[self.indices.RIGHT_EYE]

            def brow_arch_ratio(brow_pts_func):
                if len(brow_pts_func) != 5: return 0.0
                start_pt, mid_pt, end_pt = brow_pts_func[0], brow_pts_func[2], brow_pts_func[4]
                brow_length = self.get_distance(start_pt, end_pt)
                if brow_length == 0: return 0.0
                line_vec = np.array(end_pt) - np.array(start_pt)
                point_vec = np.array(mid_pt) - np.array(start_pt)
                line_len_sq = np.dot(line_vec, line_vec)
                if line_len_sq == 0: return 0.0
                proj_len = np.dot(point_vec, line_vec) / line_len_sq
                closest_pt_on_line = np.array(start_pt) + np.clip(proj_len, 0.0, 1.0) * line_vec
                arch_height = self.get_distance(mid_pt, closest_pt_on_line)
                return arch_height / brow_length if brow_length > 0 else 0.0

            features['eyebrow_arch_ratio_left'] = brow_arch_ratio(left_brow)
            features['eyebrow_arch_ratio_right'] = brow_arch_ratio(right_brow)
            features['eyebrow_slant_angle_left'] = self.get_angle_degrees(left_brow[4], left_brow[0])  # end, start
            features['eyebrow_slant_angle_right'] = self.get_angle_degrees(right_brow[0], right_brow[4])  # start, end

            left_eye_top_mid = self.get_midpoint(left_eye[1], left_eye[2])
            right_eye_top_mid = self.get_midpoint(right_eye[1], right_eye[2])
            features['eyebrow_eye_dist_left'] = self.get_distance(left_brow[2],
                                                                  left_eye_top_mid)  # Mid brow to mid eye top
            features['eyebrow_eye_dist_right'] = self.get_distance(right_brow[2], right_eye_top_mid)
            features['eyebrow_thickness_estimate'] = -1.0  # Placeholder
        except IndexError as e:
            print(f"Error: Landmark index out of bounds in eyebrow analysis: {e}")
        except Exception as e:
            print(f"Error in eyebrow analysis: {e}")
        return features

    def analyze_nose(self, landmarks, face_height_jaw_nose, ipd):
        features = {}
        if landmarks.shape[0] < max(self.indices.NOSE_BRIDGE[-1], self.indices.NOSE_TIP[-1]) + 1: return features
        try:
            nose_bridge_top = landmarks[self.indices.NOSE_BRIDGE[0]]
            nose_tip_bottom = landmarks[self.indices.NOSE_TIP[2]]  # Point 33
            nose_left_wing = landmarks[self.indices.NOSE_TIP[0]]  # Point 31
            nose_right_wing = landmarks[self.indices.NOSE_TIP[4]]  # Point 35
            nose_length = self.get_distance(nose_bridge_top, nose_tip_bottom)
            nose_width = self.get_distance(nose_left_wing, nose_right_wing)
            features['nose_length'] = nose_length
            features['nose_width'] = nose_width
            if face_height_jaw_nose is not None and face_height_jaw_nose > 0:
                features['nose_length_ratio'] = nose_length / face_height_jaw_nose
            else:
                features['nose_length_ratio'] = 0.0
            if ipd is not None and ipd > 0:
                features['nose_width_ratio'] = nose_width / ipd
            else:
                features['nose_width_ratio'] = 0.0
        except IndexError as e:
            print(f"Error: Landmark index out of bounds in nose analysis: {e}")
        except Exception as e:
            print(f"Error in nose analysis: {e}")
        return features

    def analyze_mouth(self, landmarks, face_width):
        features = {}
        if landmarks.shape[0] < max(self.indices.MOUTH_OUTER[-1],
                                    self.indices.MOUTH_INNER[-1] if self.indices.MOUTH_INNER else
                                    self.indices.MOUTH_OUTER[-1]) + 1:
            return features
        try:
            outer_mouth = landmarks[self.indices.MOUTH_OUTER]
            inner_mouth = landmarks[self.indices.MOUTH_INNER] if self.indices.MOUTH_INNER and len(
                self.indices.MOUTH_INNER) <= landmarks.shape[0] else []

            mouth_width = self.get_distance(outer_mouth[0], outer_mouth[6])  # Points 48 and 54
            features['mouth_width'] = mouth_width
            if face_width > 0:
                features['mouth_width_ratio'] = mouth_width / face_width
            else:
                features['mouth_width_ratio'] = 0.0

            upper_lip_thickness, lower_lip_thickness = -1.0, -1.0
            # Check inner_mouth points carefully based on 68-point model
            if len(inner_mouth) >= 4:  # Needs at least points 61, 62, 63 for upper lip
                # Upper lip: avg distance between (50,61), (51,62), (52,63)
                upper_dist1 = self.get_distance(outer_mouth[2], inner_mouth[1])  # 50, 61
                upper_dist2 = self.get_distance(outer_mouth[3], inner_mouth[2])  # 51, 62
                upper_dist3 = self.get_distance(outer_mouth[4], inner_mouth[3])  # 52, 63
                upper_lip_thickness = (upper_dist1 + upper_dist2 + upper_dist3) / 3.0
            else:  # Fallback if inner mouth not detailed enough (e.g. closed mouth, few points)
                upper_lip_thickness = self.get_distance(outer_mouth[3], outer_mouth[
                    9]) * 0.3  # Pts 51, 57 (center top to center bottom of outer) - rough guess

            if len(inner_mouth) >= 8:  # Needs at least points 67, 66, 65 for lower lip
                # Lower lip: avg distance between (58,67), (57,66), (56,65)
                lower_dist1 = self.get_distance(outer_mouth[10], inner_mouth[7])  # 58, 67
                lower_dist2 = self.get_distance(outer_mouth[9], inner_mouth[6])  # 57, 66
                lower_dist3 = self.get_distance(outer_mouth[8], inner_mouth[5])  # 56, 65
                lower_lip_thickness = (lower_dist1 + lower_dist2 + lower_dist3) / 3.0
            else:  # Fallback
                lower_lip_thickness = upper_lip_thickness  # very rough, or another heuristic

            features['upper_lip_thickness'] = upper_lip_thickness
            features['lower_lip_thickness'] = lower_lip_thickness
            if upper_lip_thickness > 0.1 and lower_lip_thickness > 0:  # Avoid division by zero or meaningless ratio
                features['lip_thickness_ratio'] = lower_lip_thickness / upper_lip_thickness
            else:
                features['lip_thickness_ratio'] = -1.0

            # Mouth corner angles (relative to horizontal or center)
            # Angle of line from mouth center (approx outer_mouth[3]) to corners
            mouth_center_approx_y = outer_mouth[3][1]
            left_corner_y_diff = outer_mouth[0][1] - mouth_center_approx_y
            left_corner_x_diff = outer_mouth[0][0] - outer_mouth[3][0]
            features['mouth_corner_angle_left'] = degrees(atan2(-left_corner_y_diff, left_corner_x_diff))

            right_corner_y_diff = outer_mouth[6][1] - mouth_center_approx_y
            right_corner_x_diff = outer_mouth[6][0] - outer_mouth[3][0]
            features['mouth_corner_angle_right'] = degrees(atan2(-right_corner_y_diff, right_corner_x_diff))

        except IndexError as e:
            print(f"Error: Landmark index out of bounds in mouth analysis: {e}")
        except Exception as e:
            print(f"Error in mouth analysis: {e}")
        return features

    def sample_skin_color(self, image, landmarks):
        if landmarks.shape[0] < max(self.indices.JAW[-1], self.indices.NOSE_TIP[-1], self.indices.LEFT_EYE[-1],
                                    self.indices.RIGHT_EYE[-1]) + 1:
            return "#FFFFFF"  # Default if not enough landmarks

        # Define cheek regions more robustly
        cheek_sample_regions_indices = [
            # Left Cheek (points around 2, 3, 4, 31, 40, 48) - creating a convex hull
            [self.indices.JAW[2], self.indices.JAW[3], self.indices.JAW[4], self.indices.NOSE_TIP[0],
             self.indices.LEFT_EYE[4], self.indices.MOUTH_OUTER[0]],
            # Right Cheek (points around 14, 13, 12, 35, 47, 54)
            [self.indices.JAW[14], self.indices.JAW[13], self.indices.JAW[12], self.indices.NOSE_TIP[4],
             self.indices.RIGHT_EYE[1], self.indices.MOUTH_OUTER[6]]
        ]
        all_colors, h, w = [], image.shape[0], image.shape[1]

        for region_idx_list in cheek_sample_regions_indices:
            try:
                region_pts = np.array([landmarks[i] for i in region_idx_list if i < landmarks.shape[0]], dtype=np.int32)
                if region_pts.shape[0] < 3: continue  # Need at least 3 points for a polygon

                # Create a mask for the convex hull of the points
                mask = np.zeros(image.shape[:2], np.uint8)
                cv2.drawContours(mask, [cv2.convexHull(region_pts)], -1, (255), -1, cv2.LINE_AA)

                if cv2.countNonZero(mask) == 0: continue
                mean_val = cv2.mean(image, mask=mask)  # image is BGR
                b, g, r_val = int(mean_val[0]), int(mean_val[1]), int(mean_val[2])
                # Skin color heuristic (can be refined)
                if (r_val > 95 and g > 40 and b > 20 and
                        max(r_val, g, b) - min(r_val, g, b) > 15 and
                        abs(r_val - g) > 15 and r_val > g and r_val > b):  # Basic skin filter
                    all_colors.append((r_val, g, b))  # Store as R,G,B for hex
            except Exception as e:
                print(f"Error sampling cheek region: {e}")

        if not all_colors:  # Fallback to forehead
            try:
                if landmarks.shape[0] > max(self.indices.LEFT_EYEBROW[2], self.indices.RIGHT_EYEBROW[2]):
                    # Forehead region: above eyebrows, between them
                    left_brow_center = landmarks[self.indices.LEFT_EYEBROW[2]]
                    right_brow_center = landmarks[self.indices.RIGHT_EYEBROW[2]]
                    fh_center_x = int((left_brow_center[0] + right_brow_center[0]) / 2)
                    fh_center_y = int((left_brow_center[1] + right_brow_center[1]) / 2)

                    # Estimate eyebrow height for y_offset
                    eye_dist_l = self.get_distance(landmarks[self.indices.LEFT_EYEBROW[0]],
                                                   landmarks[self.indices.LEFT_EYEBROW[4]])
                    eye_dist_r = self.get_distance(landmarks[self.indices.RIGHT_EYEBROW[0]],
                                                   landmarks[self.indices.RIGHT_EYEBROW[4]])
                    brow_height_approx = (eye_dist_l + eye_dist_r) / 4  # A fraction of brow width

                    fh_sample_y = max(0, int(fh_center_y - brow_height_approx * 1.5))  # Go up from eyebrow center

                    fh_radius = int(brow_height_approx * 0.5)  # Smaller radius for sampling
                    if fh_radius < 3: fh_radius = 3

                    fh_colors_list = []
                    for r_offset in range(-fh_radius, fh_radius + 1):
                        for c_offset in range(-fh_radius, fh_radius + 1):
                            px, py = fh_center_x + c_offset, fh_sample_y + r_offset
                            if 0 <= px < w and 0 <= py < h:
                                b_forehead, g_forehead, r_forehead = image[py, px]  # BGR
                                if (r_forehead > 95 and g_forehead > 40 and b_forehead > 20 and
                                        max(r_forehead, g_forehead, b_forehead) - min(r_forehead, g_forehead,
                                                                                      b_forehead) > 15 and
                                        abs(r_forehead - g_forehead) > 15 and r_forehead > g_forehead and r_forehead > b_forehead):
                                    fh_colors_list.append((r_forehead, g_forehead, b_forehead))
                    if fh_colors_list:
                        avg_color = np.mean(fh_colors_list, axis=0).astype(int)
                        return f"#{avg_color[0]:02X}{avg_color[1]:02X}{avg_color[2]:02X}"  # R,G,B
            except IndexError:
                print("Error: Landmark index out of bounds in forehead sampling.")
            except Exception as e:
                print(f"Error sampling forehead: {e}")
            return "#C0C0C0"  # Default grey if all fails

        avg_color = np.mean(all_colors, axis=0).astype(int)
        return f"#{avg_color[0]:02X}{avg_color[1]:02X}{avg_color[2]:02X}"  # R,G,B

    @staticmethod
    def detect_glasses_heuristic(landmarks):
        # This is a very basic heuristic and not reliable.
        # A dedicated model is needed for accurate glasses detection.
        print("Warning: Accurate glasses detection requires a dedicated model.")
        # Example Heuristic: Check if eyebrow-to-eye distance is unusually large,
        # or if there's a consistent line of landmarks missing/occluded around eyes.
        # For now, returning False.
        return False

    def extract_all_features(self, image_cv2_original):
        if not self.models_loaded:
            return {"error": "Models not loaded in extractor", "details": "Cannot process image."}

        image_cv2 = image_cv2_original.copy()

        if len(image_cv2.shape) == 2:  # Grayscale image
            print("Input image is grayscale. Converting to BGR for consistency.")
            image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_GRAY2BGR)

        if image_cv2.shape[2] == 4:  # BGRA or RGBA
            print("Image has an alpha channel. Converting to BGR.")
            image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGRA2BGR)

        if image_cv2.dtype != np.uint8:
            print(f"Image dtype is {image_cv2.dtype}, attempting conversion to uint8.")
            if np.issubdtype(image_cv2.dtype, np.floating):  # Float to uint8
                if image_cv2.max() <= 1.0:  # Assuming 0-1 range for float
                    image_cv2 = (image_cv2 * 255).astype(np.uint8)
                else:  # Assuming 0-255 range for float already
                    image_cv2 = image_cv2.astype(np.uint8)
            elif np.issubdtype(image_cv2.dtype, np.integer):  # Other integer types
                image_cv2 = image_cv2.astype(np.uint8)  # Direct cast, might clip
            else:
                return {"error": "Unsupported image data type",
                        "details": f"Cannot convert {image_cv2.dtype} to uint8."}

        image_for_detection = image_cv2  # Default to color for MMOD
        if not isinstance(self.detector, dlib.cnn_face_detection_model_v1):
            # HOG detector needs grayscale
            gray_for_hog = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
            image_for_detection = gray_for_hog

        try:
            faces = self.detector(image_for_detection, 1)  # Upsample by 1
        except RuntimeError as e:
            print(f"Dlib runtime error during face detection: {e}")
            return {"error": "Face detection failed (dlib runtime)", "details": str(e)}
        except Exception as e:
            print(f"Generic error during face detection: {e}")
            return {"error": "Face detection failed", "details": str(e)}

        if len(faces) == 0:
            return {"error": "No faces detected"}
        if len(faces) > 1:
            print("Warning: Multiple faces detected, using the first one.")

        face = faces[0]
        rect = face.rect if isinstance(face, dlib.mmod_rectangle) else face

        try:
            # Predictor works well with color (BGR) image
            shape = self.predictor(image_cv2, rect)
            landmarks = self.landmarks_to_np(shape)
            if landmarks.shape[0] != 68:
                return {"error": "Landmark detection failed", "details": "Incorrect number of landmarks"}
        except RuntimeError as e:
            print(f"Dlib runtime error during landmark prediction: {e}")
            return {"error": "Landmark prediction failed (dlib runtime)", "details": str(e)}
        except Exception as e:
            print(f"Error during landmark prediction: {e}")
            return {"error": "Landmark prediction failed", "details": str(e)}

        all_features = {}
        try:
            face_props = self.analyze_face_proportions(landmarks)
            all_features.update(face_props)
            face_width = face_props.get('face_width')  # Can be None if not calculated
            face_height_jaw_nose = face_props.get('face_height_jaw_nose')  # Can be None

            eye_features = self.analyze_eyes(landmarks, face_width if face_width else 0)
            all_features.update(eye_features)
            ipd = eye_features.get('interpupillary_distance')  # Can be None

            all_features.update(self.analyze_eyebrows(landmarks))
            all_features.update(
                self.analyze_nose(landmarks, face_height_jaw_nose if face_height_jaw_nose else 0, ipd if ipd else 0))
            all_features.update(self.analyze_mouth(landmarks, face_width if face_width else 0))

            # image_cv2 is BGR uint8 at this point
            all_features['skin_color_rgb_hex'] = self.sample_skin_color(image_cv2, landmarks)
            all_features['has_glasses_heuristic'] = self.detect_glasses_heuristic(landmarks)

            serializable_features = {}
            for key, value in all_features.items():
                if isinstance(value, (
                np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
                np.uint64)):
                    serializable_features[key] = int(value)
                elif isinstance(value, (np.float_, np.float16, np.float32, np.float64)):
                    serializable_features[key] = float(value) if not (np.isnan(value) or np.isinf(value)) else None
                elif isinstance(value, (np.ndarray,)):
                    serializable_features[key] = value.tolist()
                elif isinstance(value, (bool, np.bool_)):
                    serializable_features[key] = bool(value)
                else:
                    serializable_features[key] = value
            return serializable_features
        except Exception as e:
            print(f"Error during feature analysis sub-step: {e}")
            import traceback
            traceback.print_exc()
            return {"error": "Feature analysis failed during sub-steps", "details": str(e)}


if __name__ == '__main__':
    # Example Usage (for testing the class directly)
    DLIB_FACE_DETECTOR_PATH_TEST = "mmod_human_face_detector.dat"  # or "" or None for HOG
    DLIB_LANDMARK_PREDICTOR_PATH_TEST = "shape_predictor_68_face_landmarks.dat"
    DEFAULT_IMAGE_PATH_TEST = "sad.png"  # Replace with your test image

    if not os.path.exists(DLIB_LANDMARK_PREDICTOR_PATH_TEST):
        print(f"Test Error: Landmark predictor not found at {DLIB_LANDMARK_PREDICTOR_PATH_TEST}")
        exit()
    if DLIB_FACE_DETECTOR_PATH_TEST and not DLIB_FACE_DETECTOR_PATH_TEST.strip() == "" and not os.path.exists(
            DLIB_FACE_DETECTOR_PATH_TEST):
        print(
            f"Test Warning: MMOD Face detector specified ('{DLIB_FACE_DETECTOR_PATH_TEST}') but not found. Extractor will attempt HOG fallback.")

    extractor = FacialFeatureExtractor(
        face_detector_path=DLIB_FACE_DETECTOR_PATH_TEST,
        landmark_predictor_path=DLIB_LANDMARK_PREDICTOR_PATH_TEST
    )

    if extractor.models_loaded:
        if not os.path.exists(DEFAULT_IMAGE_PATH_TEST):
            print(f"Test Error: Image not found at {DEFAULT_IMAGE_PATH_TEST}")
            exit()

        image = cv2.imread(DEFAULT_IMAGE_PATH_TEST, cv2.IMREAD_UNCHANGED)  # Read with alpha if present
        if image is None:
            print(f"Test Error: Could not load image {DEFAULT_IMAGE_PATH_TEST}")
            exit()

        print(f"Test Image shape: {image.shape}, dtype: {image.dtype}")
        features = extractor.extract_all_features(image)

        if "error" in features:
            print(f"Test Error extracting features: {features}")
        else:
            print("\n--- Extracted Features (Class Test) ---")
            for key, value in features.items():
                if isinstance(value, float):
                    print(f"  - {key}: {value:.4f}")
                else:
                    print(f"  - {key}: {value}")
            print("------------------------------------")
    else:
        print("Test Error: Models could not be loaded in FacialFeatureExtractor. Cannot run test.")