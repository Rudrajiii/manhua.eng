import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import asyncio
import easyocr
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from constants import SKIP_WORDS , DEFAULT_FONT , FONT_CANDIDATES , ITALIC_CANDIDATES
from llm import pipe_to_llm
import logging

logger = logging.getLogger(__name__)

# Ensure SKIP_WORDS are lowercase for case-insensitive matching
SKIP_WORDS = frozenset(w.lower() for w in SKIP_WORDS)

class DetectTexts:
    def __init__(self , client_os:str , image_path:str , output_folder:str):
        self.client_os = client_os
        self.image_path = image_path
        self.output_folder = output_folder

        if self.client_os == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    def cvt2pil(self , cv_image):
        image = Image.fromarray(cv2.cvtColor(cv_image , cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        return [image , draw]

    def should_skip(self, text: str) -> bool:
        t = text.lower()
        return any(w in t for w in SKIP_WORDS)
    
    def get_font_path(self , font_style) -> str:
        if font_style == "ITALIC":
            for font_file in ITALIC_CANDIDATES:
                if Path(font_file).exists():
                    return font_file
            return DEFAULT_FONT
        else:
            for font_file in FONT_CANDIDATES:
                if Path(font_file).exists():
                    return font_file
            return DEFAULT_FONT
    
    def get_filtered_text_bubbles_and_draw_text_detections_with_index_for_each_bubble(self, bubble_map , bubble_text_map , draw , font_path):
        # Draw text detections with index for each bubble
        for bubble_idx, detections in reversed(bubble_text_map.items()):
            if not detections:
                continue
            
            if bubble_idx == -1:
                continue
            logger.debug(f"\nBubble {bubble_idx}:")
            
            bubble_map[bubble_idx] = []
            # Index starts from 0 for each bubble
            for idx, detection in enumerate(detections):
                text = detection[1]
                bbox = detection[0]
                
                logger.debug(f"[{idx}] {text}")
                bubble_map[bubble_idx].append(text.strip())
                
                # Get bounding box
                points = np.array([(int(point[0]), int(point[1])) for point in bbox])
                x_min, y_min = points.min(axis=0)
                x_max, y_max = points.max(axis=0)
                
                # Draw red border around text
                draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=3)
                
                # Draw index number at top-left corner
                try:
                    index_font = ImageFont.truetype(font_path, 20)
                except:
                    index_font = ImageFont.load_default()
                
                # Draw index with white background for visibility
                index_text = f"[{idx}]"
                index_bbox = index_font.getbbox(index_text)
                index_width = index_bbox[2] - index_bbox[0]
                index_height = index_bbox[3] - index_bbox[1]
                
                # White background
                draw.rectangle([x_min, y_min - index_height - 5, x_min + index_width + 10, y_min], 
                            fill='white', outline='red', width=2)
                # Red index text
                draw.text((x_min + 5, y_min - index_height - 3), index_text, font=index_font, fill='red')
        
        return bubble_map

    def text_detected_image(self , image , bubble_text_map):
        output_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        input_filename = Path(self.image_path).name
        output_path = Path(self.output_folder) / input_filename
        cv2.imwrite(str(output_path), output_image)
        logger.debug(f"\n✓ Output image saved to: {output_path}")
        logger.debug(f"✓ Total speech bubbles: {len([v for v in bubble_text_map.values() if v])}")
        return output_path
    
    async def translate_detected_texts(self , translations , filtered_bubble_map ):
        for bubble_idx, text_arr in filtered_bubble_map.items():
            text = " ".join(text_arr)
            translated_text = await pipe_to_llm(text)
            print(f"Bubble {bubble_idx} - Translated: {translated_text}")
            translations[bubble_idx] = translated_text
        return translations

    def draw_image_with_translated_text_overlaid(self , cv_image , bubble_text_map , translations):
        translation_output_folder = Path(self.output_folder).parent / 'translations'
        translation_output_folder.mkdir(exist_ok=True)

        italic_font_path = self.get_font_path("ITALIC")

        inpaint_mask = np.zeros((cv_image.shape[0], cv_image.shape[1]), dtype=np.uint8)
        for bubble_idx, detections in bubble_text_map.items():
            if bubble_idx == -1:
                continue
                
            if not detections:
                continue
            
            # Mark text regions in the mask
            for detection in detections:
                bbox = detection[0]
                points = np.array([(int(point[0]), int(point[1])) for point in bbox])
                x_min, y_min = points.min(axis=0)
                x_max, y_max = points.max(axis=0)
                
                # Expand the region slightly to ensure full coverage
                padding = 3
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(cv_image.shape[1], x_max + padding)
                y_max = min(cv_image.shape[0], y_max + padding)
                
                # Fill mask region (white = inpaint this area)
                cv2.rectangle(inpaint_mask, (x_min, y_min), (x_max, y_max), 255, -1)
        
        # Apply inpainting to remove original text
        cv_inpainted = cv2.inpaint(cv_image, inpaint_mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
        
        # Convert inpainted image to PIL for drawing translations
        clean_image, draw_translation = self.cvt2pil(cv_inpainted)

        for bubble_idx, detections in bubble_text_map.items():
            if bubble_idx == -1 or bubble_idx not in translations:
                continue
                
            if not detections:
                continue
            
            # Get bubble bounding box
            all_points = []
            for detection in detections:
                points = np.array(detection[0])
                all_points.extend(points)
            
            all_points = np.array(all_points)
            bubble_x_min, bubble_y_min = all_points.min(axis=0)
            bubble_x_max, bubble_y_max = all_points.max(axis=0)
            bubble_center_x = (bubble_x_min + bubble_x_max) / 2
            bubble_center_y = (bubble_y_min + bubble_y_max) / 2
            bubble_width = bubble_x_max - bubble_x_min
            bubble_height = bubble_y_max - bubble_y_min
            
            # Get translated text
            translated_text = translations[bubble_idx]
            
            # Skip if translation is empty
            if not translated_text or translated_text.strip() == "":
                logger.debug(f"Skipping bubble {bubble_idx} - empty translation")
                continue
            
            # Normalize special characters that might not render in fonts
            # Replace various dash types with regular hyphen
            translated_text = translated_text.replace('‑', '-')  # non-breaking hyphen
            translated_text = translated_text.replace('–', '-')  # en dash
            translated_text = translated_text.replace('—', '-')  # em dash
            translated_text = translated_text.replace('−', '-')  # minus sign
            # Replace curly quotes with straight quotes
            translated_text = translated_text.replace(''', "'")
            translated_text = translated_text.replace(''', "'")
            translated_text = translated_text.replace('"', '"')
            translated_text = translated_text.replace('"', '"')
            
            # Start with a reasonable font size and adjust down if text doesn't fit
            max_font_size = min(int(bubble_height / 3), 36)
            min_font_size = 30
            
            # Try different font sizes until text fits
            best_font = None
            best_lines = None
            
            for font_size in range(max_font_size, min_font_size - 1, -2):
                try:
                    test_font = ImageFont.truetype(italic_font_path, font_size)
                except:
                    test_font = ImageFont.load_default()
                    break
                
                # Wrap text with current font size
                words = translated_text.split()
                lines = []
                current_line = []
                
                for word in words:
                    test_line = ' '.join(current_line + [word])
                    bbox = test_font.getbbox(test_line)
                    text_width = bbox[2] - bbox[0]
                    
                    # Use 85% of bubble width to ensure padding
                    if text_width < bubble_width * 0.85:
                        current_line.append(word)
                    else:
                        if current_line:
                            lines.append(' '.join(current_line))
                            current_line = [word]
                        else:
                            # Single word too long, force it
                            lines.append(word)
                            current_line = []
                
                if current_line:
                    lines.append(' '.join(current_line))
                
                # Calculate total height needed
                line_height = font_size + 8
                total_height = len(lines) * line_height
                
                # Check if it fits within 85% of bubble height
                if total_height <= bubble_height * 0.85:
                    best_font = test_font
                    best_lines = lines
                    break
            
            # Fallback if no size worked
            if best_font is None:
                try:
                    best_font = ImageFont.truetype(italic_font_path, min_font_size)
                except:
                    best_font = ImageFont.load_default()
                
                # Do one more wrap with minimum font
                words = translated_text.split()
                best_lines = []
                current_line = []
                
                for word in words:
                    test_line = ' '.join(current_line + [word])
                    bbox = best_font.getbbox(test_line)
                    text_width = bbox[2] - bbox[0]
                    
                    if text_width < bubble_width * 0.85:
                        current_line.append(word)
                    else:
                        if current_line:
                            best_lines.append(' '.join(current_line))
                        current_line = [word]
                
                if current_line:
                    best_lines.append(' '.join(current_line))
            
            trans_font = best_font
            lines = best_lines
            
            # Draw each line centered in the bubble
            # Get actual line height from font
            sample_bbox = trans_font.getbbox("Ay")
            line_height = (sample_bbox[3] - sample_bbox[1]) + 8
            total_text_height = len(lines) * line_height
            start_y = bubble_center_y - total_text_height / 2
            
            for i, line in enumerate(lines):
                bbox = trans_font.getbbox(line)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                text_x = bubble_center_x - text_width / 2
                text_y = start_y + i * line_height
                
                # Draw translated text in italic
                draw_translation.text((text_x, text_y), line, font=trans_font, fill='black')

        translation_output = cv2.cvtColor(np.array(clean_image), cv2.COLOR_RGB2BGR)
        input_filename = Path(self.image_path).name
        translation_path = translation_output_folder / input_filename
        cv2.imwrite(str(translation_path), translation_output)
        logger.debug(f"\n✓ Translated image saved to: {translation_path}")

    def detect_speech_bubbles(self , cv_image):
        """
        Detect speech bubble shapes (white regions) using contour detection.
        Returns bounding boxes for each speech bubble.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Use lower threshold to catch bubbles that aren't pure white
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to close gaps
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours - use RETR_TREE to detect nested bubbles
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and aspect ratio
        speech_bubbles = []
        image_area = cv_image.shape[0] * cv_image.shape[1]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            # Skip very small areas and areas that are too large (likely the entire background)
            if area > 2000 and area < image_area * 0.8:
                x, y, w, h = cv2.boundingRect(contour)
                speech_bubbles.append((x, y, x + w, y + h))
        
        # Remove duplicate/overlapping bubbles by keeping the larger ones
        filtered_bubbles = []
        for bubble in speech_bubbles:
            x1, y1, x2, y2 = bubble
            is_duplicate = False
            for existing in filtered_bubbles:
                ex1, ey1, ex2, ey2 = existing
                # Check if this bubble is mostly contained in an existing one
                overlap_x = max(0, min(x2, ex2) - max(x1, ex1))
                overlap_y = max(0, min(y2, ey2) - max(y1, ey1))
                overlap_area = overlap_x * overlap_y
                bubble_area = (x2 - x1) * (y2 - y1)
                if overlap_area > bubble_area * 0.8:
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered_bubbles.append(bubble)
        
        return filtered_bubbles

    def group_text_by_speech_bubbles(self , filtered_results , speech_bubbles):
        """
        Group text detections based on which speech bubble they fall within.
        Returns dict with bubble_idx -> list of detections
        """
        if not filtered_results:
            return {}
        
        # Map each text detection to a speech bubble
        bubble_text_map = {i: [] for i in range(len(speech_bubbles))}
        unassigned = []
        
        for detection in filtered_results:
            points = np.array(detection[0])
            center_x = points[:, 0].mean()
            center_y = points[:, 1].mean()
            
            # Find which speech bubble contains this text
            assigned = False
            for bubble_idx, (bx_min, by_min, bx_max, by_max) in enumerate(speech_bubbles):
                if bx_min <= center_x <= bx_max and by_min <= center_y <= by_max:
                    bubble_text_map[bubble_idx].append(detection)
                    assigned = True
                    break
            
            if not assigned:
                unassigned.append(detection)
        
        # Split large bubbles into sub-bubbles based on spatial clustering
        final_bubble_map = {}
        current_bubble_idx = 0
        
        for bubble_idx, detections in bubble_text_map.items():
            if not detections:
                continue
                
            # If bubble has many texts, try to split into sub-bubbles
            if len(detections) > 3:
                sub_bubbles = self.split_bubble_by_clustering(detections)
                for sub_bubble in sub_bubbles:
                    final_bubble_map[current_bubble_idx] = sub_bubble
                    current_bubble_idx += 1
            else:
                final_bubble_map[current_bubble_idx] = detections
                current_bubble_idx += 1
        
        # Add unassigned as a separate group
        if unassigned:
            final_bubble_map[-1] = unassigned
            
        logger.debug(
            "Bubble Text Map",
            final_bubble_map
        )
        return final_bubble_map
    
    def split_bubble_by_clustering(self , detections):
        """
        Split a large bubble into sub-bubbles based on spatial clustering.
        Uses vertical distance between consecutive detections
        WITHOUT sorting (preserves original order).
        """
        if len(detections) <= 1:
            return [detections]

        # Compute Y center for each detection (original order)
        y_coords = []
        for detection in detections:
            points = np.array(detection[0])
            y_coords.append(points[:, 1].mean())

        sub_bubbles = []
        current_group = [detections[0]]

        for i in range(1, len(detections)):
            y_diff = abs(y_coords[i] - y_coords[i - 1])

            # If gap is large, start a new sub-bubble
            if y_diff > 80:
                sub_bubbles.append(current_group)
                current_group = [detections[i]]
            else:
                current_group.append(detections[i])

        # Add the last group
        sub_bubbles.append(current_group)

        return sub_bubbles
    
    async def detect_and_draw_text(self):
        Path(self.output_folder).mkdir(exist_ok=True)

        reader = easyocr.Reader(['ch_sim', 'en'])
        cv_image = cv2.imread(self.image_path)

        if cv_image is None:
            logger.error(f"Error: Could not read image from {self.image_path}")
            return
        
        # convert to PIL image
        # image = Image.fromarray(cv2.cvtColor(cv_image , cv2.COLOR_BGR2RGB))
        # draw = ImageDraw.draw(image)

        image , draw = self.cvt2pil(cv_image)

        logger.debug("Detecting text...")
        detections = reader.readtext(self.image_path)

        logger.debug("Detecting speech bubbles...")
        speech_bubbles = self.detect_speech_bubbles(cv_image)
        logger.debug(f"Found {len(speech_bubbles)} speech bubbles")

        # lets filter out watermarks
        filtered_results = []

        for detection in detections:
            text = detection[1]

            normalized_text = text.lower().replace(' ', '').replace('.', '').replace(':', '').replace(";" , ",")
            skiped = self.should_skip(normalized_text)
            if not skiped:
                filtered_results.append(detection)

        # Group text by speech bubbles
        logger.debug("Grouping text by speech bubbles...")
        bubble_text_map = self.group_text_by_speech_bubbles(filtered_results, speech_bubbles)

        logger.debug(f"Original detections: {len(detections)}, After filtering: {len(filtered_results)}\n")
        font_path = self.get_font_path("NORMAL")

        filtered_bubble_map = self.get_filtered_text_bubbles_and_draw_text_detections_with_index_for_each_bubble({} , bubble_text_map , draw , font_path)
        self.text_detected_image(image , bubble_text_map)
        eng_translations = await self.translate_detected_texts({} , filtered_bubble_map)
        self.draw_image_with_translated_text_overlaid(cv_image , bubble_text_map , eng_translations)


if __name__ == "__main__":
    manhua_panel = DetectTexts("win32" , "manhua/i_4.png" , "my_output")
    asyncio.run(manhua_panel.detect_and_draw_text())