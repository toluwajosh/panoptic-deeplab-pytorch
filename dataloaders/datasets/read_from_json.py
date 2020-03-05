"""Read annotation json
As in citiscapes dataset
"""
import json


def polygon_to_bbox(polygon):
    min_x, min_y = 999999, 999999
    max_x, max_y = 0, 0
    for point in polygon:
        x, y = point
        if x < min_x:
            min_x = x
        if x > max_x:
            max_x = x
        if y < min_y:
            min_y = y
        if y > max_y:
            max_y = y
    return {"xmin": min_x, "xmax": max_x, "ymin": min_y, "ymax": max_y}


def load_json_data(json_file):
    with open(json_file, "r") as file:
        data = json.load(file)
    annotations = []
    for object_data in data["objects"]:
        label = object_data["label"]
        if label == "out of roi":
            continue
        # labels.append(object_data["label"])
        polygon = object_data["polygon"]
        # convert polygon to bounding box
        annotations.append(
            {"label": label,
             "bbox": polygon_to_bbox(polygon)}
        )
    return annotations


if __name__ == "__main__":
    load_json_data("frankfurt_000000_000294_gtFine_polygons.json")
