import cv2
import numpy as np
import json
import os
from pathlib import Path

"""
Interactive Lane Drawing Tool
==============================
Usage: python draw_lanes.py

Instructions:
1. Left click: Add points to current lane polygon
2. Right click: Complete current lane and start next lane
3. Press 'u': Undo last point
4. Press 'c': Clear current lane
5. Press 'r': Reset all lanes
6. Press 's': Save colored reference image
7. Press 'q': Quit without saving
8. Press ESC: Quit without saving

Each lane will be filled with a different color automatically.
"""

class LaneDrawer:
    def __init__(self, image_path, video_name):
        self.image_path = image_path
        self.video_name = video_name
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")
        
        self.display_image = self.original_image.copy()
        self.colored_image = self.original_image.copy()
        
        self.lanes = []  # List of lanes, each lane is a list of points
        self.current_lane = []  # Points for the current lane being drawn
        self.current_lane_idx = 0
        
        # Predefined colors for each lane (BGR format)
        self.lane_colors = [
            (255, 0, 0),      # Blue
            (0, 255, 0),      # Green
            (0, 0, 255),      # Red
            (255, 255, 0),    # Cyan
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Yellow
            (128, 0, 255),    # Purple
            (255, 128, 0),    # Orange
            (0, 128, 255),    # Light Blue
            (128, 255, 0),    # Light Green
        ]
        
        self.window_name = "Lane Drawing Tool"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        self.alpha = 0.0  # Transparency for overlay (0.0 = fully opaque, 1.0 = fully transparent)
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add point to current lane
            self.current_lane.append((x, y))
            self.update_display()
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Complete current lane and start next
            if len(self.current_lane) >= 3:  # Need at least 3 points for a polygon
                self.lanes.append(self.current_lane.copy())
                self.fill_lane(len(self.lanes) - 1)
                self.current_lane = []
                self.current_lane_idx += 1
                print(f"Lane {len(self.lanes)} completed. Starting Lane {len(self.lanes) + 1}...")
                self.update_display()
            else:
                print("Need at least 3 points to complete a lane!")
    
    def fill_lane(self, lane_idx):
        """Fill the specified lane with its color"""
        if lane_idx >= len(self.lanes):
            return
        
        points = np.array(self.lanes[lane_idx], dtype=np.int32)
        color = self.lane_colors[lane_idx % len(self.lane_colors)]
        
        if self.alpha == 0.0:
            # Fully opaque - direct fill without blending
            cv2.fillPoly(self.colored_image, [points], color)
        else:
            # Create overlay with transparency
            overlay = self.colored_image.copy()
            cv2.fillPoly(overlay, [points], color)
            
            # Blend with original (alpha for overlay, 1-alpha for original)
            cv2.addWeighted(overlay, self.alpha, self.colored_image, 1 - self.alpha, 0, self.colored_image)
    
    def update_display(self):
        """Update the display image with current drawing state"""
        self.display_image = self.colored_image.copy()
        
        # Draw completed lanes with outlines
        for i, lane in enumerate(self.lanes):
            points = np.array(lane, dtype=np.int32)
            color = self.lane_colors[i % len(self.lane_colors)]
            cv2.polylines(self.display_image, [points], True, color, 2)
            
            # Draw lane number
            if len(lane) > 0:
                center = np.mean(points, axis=0).astype(int)
                cv2.putText(self.display_image, f"L{i+1}", tuple(center), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Draw current lane being edited
        if len(self.current_lane) > 0:
            for i, point in enumerate(self.current_lane):
                cv2.circle(self.display_image, point, 5, (0, 255, 255), -1)
                if i > 0:
                    cv2.line(self.display_image, self.current_lane[i-1], point, (0, 255, 255), 2)
            
            # Draw line from last point to first to show closure
            if len(self.current_lane) > 1:
                cv2.line(self.display_image, self.current_lane[-1], self.current_lane[0], 
                        (0, 255, 255), 1, cv2.LINE_AA)
        
        # Draw instructions
        instructions = [
            f"Lane {len(self.lanes) + 1} | Points: {len(self.current_lane)}",
            "Left: Add point | Right: Complete lane",
            "u: Undo | c: Clear | r: Reset | s: Save | q/ESC: Quit"
        ]
        
        y_offset = 30
        for i, text in enumerate(instructions):
            cv2.putText(self.display_image, text, (10, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(self.display_image, text, (10, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        cv2.imshow(self.window_name, self.display_image)
    
    def undo_last_point(self):
        """Remove the last point from current lane"""
        if len(self.current_lane) > 0:
            self.current_lane.pop()
            print(f"Undone. Points remaining: {len(self.current_lane)}")
            self.update_display()
    
    def clear_current_lane(self):
        """Clear all points in current lane"""
        self.current_lane = []
        print("Current lane cleared.")
        self.update_display()
    
    def reset_all(self):
        """Reset all lanes and start over"""
        self.lanes = []
        self.current_lane = []
        self.current_lane_idx = 0
        self.colored_image = self.original_image.copy()
        print("All lanes reset.")
        self.update_display()
    
    def save_output(self):
        """Save the colored reference image and lane data"""
        base_dir = Path(self.image_path)
        output_dir = base_dir / f"masks_{self.video_name}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save colored reference image
        colored_ref_path = output_dir / f"{self.video_name}_colored_ref.jpg"
        cv2.imwrite(str(colored_ref_path), self.colored_image)
        print(f"✓ Colored reference image saved: {colored_ref_path}")
        
        # Save lane coordinates as JSON
        lane_data = {
            "video_name": self.video_name,
            "num_lanes": len(self.lanes),
            "lanes": [
                {
                    "lane_id": i + 1,
                    "points": lane,
                    "color": self.lane_colors[i % len(self.lane_colors)]
                }
                for i, lane in enumerate(self.lanes)
            ]
        }
        
        json_path = output_dir / "lane_data.json"
        with open(json_path, 'w') as f:
            json.dump(lane_data, f, indent=2)
        print(f"✓ Lane data saved: {json_path}")
        
        print(f"\n=== Summary ===")
        print(f"Total lanes drawn: {len(self.lanes)}")
        print(f"Output directory: {output_dir}")
        print(f"\nNext step: Use _make_mask.py to generate masks from the colored reference image.")
    
    def run(self):
        """Main loop"""
        print("\n" + "="*60)
        print("Lane Drawing Tool Started")
        print("="*60)
        print(f"Image: {self.image_path}")
        print(f"Video: {self.video_name}")
        print("\nInstructions:")
        print("  - Left click: Add points to define lane boundary")
        print("  - Right click: Complete current lane and start next")
        print("  - Press 'u': Undo last point")
        print("  - Press 'c': Clear current lane")
        print("  - Press 'r': Reset all lanes")
        print("  - Press 's': Save colored reference image")
        print("  - Press 'q' or ESC: Quit")
        print("="*60)
        print("\nStarting Lane 1...")
        
        self.update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('u'):
                self.undo_last_point()
            
            elif key == ord('c'):
                self.clear_current_lane()
            
            elif key == ord('r'):
                confirm = input("\nReset all lanes? (y/n): ")
                if confirm.lower() == 'y':
                    self.reset_all()
            
            elif key == ord('s'):
                if len(self.lanes) == 0:
                    print("No lanes to save! Draw at least one lane first.")
                else:
                    self.save_output()
                    print("\nPress any key to continue or 'q' to quit...")
            
            elif key == ord('q') or key == 27:  # 'q' or ESC
                print("\nQuitting without saving...")
                break
        
        cv2.destroyAllWindows()


def main():
    # Configuration
    video_name = "Asheque Rahman - Export_ T4 _ Arrivals _ 1.28.2025 _ 3 pm - 4 pm"
    base_dir = Path(r"c:\Users\leo\Desktop\jfk")
    image_path = base_dir / "masks" / "ref_imgs" / f"{video_name}_ref.jpg"
    
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        print("\nAvailable reference images:")
        ref_dir = base_dir / "masks" / "ref_imgs"
        if ref_dir.exists():
            for img in sorted(ref_dir.glob("*_ref.jpg")):
                print(f"  - {img.name}")
        return
    
    drawer = LaneDrawer(str(image_path), video_name)
    drawer.run()


if __name__ == "__main__":
    main()
