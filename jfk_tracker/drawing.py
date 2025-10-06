import cv2
import numpy as np

def draw_lane_overlay(frame, lane_contours, lane_colors, alpha=0.3):
    overlay = frame.copy()
    for ln, contours in lane_contours.items():
        color = lane_colors.setdefault(ln, np.random.randint(50,256,(3,)).tolist())
        cv2.drawContours(overlay, contours, -1, color, thickness=cv2.FILLED)
    return cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)

def draw_detections(frame, frame_idx, fps,
                    boxes, tids, cats, lanes,
                    last_move_by_id, dwell_secs, dist_thresh_px):
    """
    Draw orange/red boxes for dwellers; green dot otherwise.
    Also updates last_move_by_id: id -> (last_move_frame, (x,y))
    """
    for i, b in enumerate(boxes):
        tid, cat, ln = tids[i], cats[i], lanes[i]
        x, y, w, h = b
        center = (int(x), int(y))

        if cat != "people":
            cur = (int(x), int(y))
            if tid not in last_move_by_id:
                last_move_by_id[tid] = (frame_idx, cur)

            last_frame, last_pos = last_move_by_id[tid]
            moved = np.linalg.norm(np.array(cur) - np.array(last_pos)) > dist_thresh_px
            if moved:
                last_move_by_id[tid] = (frame_idx, cur)

            dt = (frame_idx - last_move_by_id[tid][0]) / fps
            if dt >= dwell_secs:
                # orange until 2x threshold, then red
                color = (0,165,255) if dt < 2*dwell_secs else (0,0,255)
                tl = (int(x - w/2), int(y - h/2))
                br = (int(x + w/2), int(y + h/2))
                cv2.rectangle(frame, tl, br, color, 2)
                cv2.putText(frame, f"{dt:.1f}s", (tl[0], tl[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                cv2.circle(frame, center, 3, (0,255,0), -1)
        else:
            cv2.circle(frame, center, 3, (0,255,0), -1)

def compose_text_area(frame_width, text_area_height, lane_contours, lane_counts, lane_over, categories):
    # HxWx3 (BGR)
    text_area = np.zeros((text_area_height, frame_width, 3), dtype=np.uint8)

    font, scale, color, line_type = cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1
    x0, y0, lh = 10, 30, 25
    line = 0
    for ln in sorted(lane_contours.keys(), key=lambda x: str(x)):
        parts = [f"Lane {ln}:"]
        for cat in categories:
            parts.append(f"{cat}: {lane_counts[ln].get(cat, 0)}")
        parts.append(f"Over threshold: {len(lane_over.get(ln, set()))}")
        txt = "  ".join(parts)
        y = y0 + line * lh
        cv2.putText(text_area, txt, (x0, y), font, scale, color, line_type)
        line += 1
    return text_area  # HxWx3

