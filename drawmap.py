# draw map usinf opencv and paint objects as rectangles
import cv2
import numpy as np

objects_as_points = []
def register_object(
        origin_x: float,
        origin_y: float,
        width: float,
        height: float
):
    objects_as_points.append((origin_x, origin_y, width, height))

def register_map_for_kitchen():
    # register objects in the map as rectangles defined by (origin_x, origin_y, width, height)
    # register_object(0, 0, 368, 470)  # Outer walls
    register_object(264, 0, 104, 104)  # doorway to bathroom
    register_object(264 - 120, 0, 120, 40)  # desk
    register_object(0, 88 + 88, 75, 121)  # table
    register_object(368 - 62, 104 + 80, 62, 230)  # counter under window
    register_object(368 - 147 - 62, 470 - 62, 209, 62)  # counter under cabinet
    register_object(368 - 147 - 62 - 11 - 50, 470 - 60, 50, 60)  # cooker
    register_object(368 - 147 - 62 - 11 - 50 - 33 - 54, 470 - 54, 54, 54)  # fridge


if __name__ == "__main__":
    register_map_for_kitchen()
    map_size = (470, 368)  # (height, width) in cm

    map_image = np.zeros((map_size[0], map_size[1], 3), dtype=np.uint8)
    for obj in objects_as_points:
        origin_x, origin_y, width, height = obj
        # cv2.putText(map_image, f"{center_x:.1f},{center_y:.1f}", (int(origin_y), int(origin_x + height / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.rectangle(map_image, (int(origin_x), int(origin_y)), (int(origin_x + width), int(origin_y + height)), (255, 255, 255), -1)
    # cv2.imwrite("map.png", map_image)
    # invert map on the y-axis so that (0, 0) is bottom left and (width, height) is top right
    map_image = cv2.flip(map_image, 0)
    cv2.imshow("map", map_image)
    cv2.waitKey(0)

