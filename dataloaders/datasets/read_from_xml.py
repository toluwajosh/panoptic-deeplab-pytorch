# https://docs.python.org/2/library/xml.etree.elementtree.html#treebuilder-objects

import xml.etree.ElementTree as ET
from collections import defaultdict


def etree_to_dict(t):
    """parse an xml tree root to dictionary
    Following the answer here: https://stackoverflow.com/questions/2148119/how-to-convert-an-xml-string-to-a-dictionary
    Arguments:
        t {[xml.etree.ElementTree Element]} -- [Root to an xml tree]

    Returns:
        [Dict] -- [xml tree converted to a dictionary]
    """
    d = {t.tag: {} if t.attrib else None}
    children = list(t)
    if children:
        dd = defaultdict(list)
        for dc in map(etree_to_dict, children):
            for k, v in dc.items():
                dd[k].append(v)
        d = {t.tag: {k: v[0] if len(v) == 1 else v for k, v in dd.items()}}
    # print(d)
    if t.attrib:
        d[t.tag].update(("@" + k, v) for k, v in t.attrib.items())
    if t.text:
        text = t.text.strip()
        if children or t.attrib:
            if text:
                d[t.tag]["#text"] = text
        else:
            d[t.tag] = text
    return d


def parse_object_bbox(tree_xml_path):
    """Parse an xml annotation to objects bounding box

    Arguments:
        tree_xml_path {str} -- xml file path

    Returns:
        list -- list of bounding box elements
    """
    tree_root = ET.parse(tree_xml_path).getroot()
    tree_dict = etree_to_dict(tree_root)
    data_object = tree_dict["annotation"]["object"]
    bbox_list = []
    if isinstance(data_object, dict):
        for bb_key in data_object:
            if bb_key == "bndbox":
                bbox_list.append(data_object[bb_key])
    else:
        for object_instance in data_object:
            bbox_list.append(object_instance["bndbox"])
    return bbox_list


if __name__ == "__main__":
    from pprint import pprint

    # An example of function usage
    # for the case of single and multiple objects
    # tree_root = ET.parse("dataloaders/datasets/2008_007069.xml").getroot()
    tree_root = ET.parse("dataloaders/datasets/2007_000027.xml").getroot()

    data_dict = etree_to_dict(tree_root)
    data_object = data_dict["annotation"]["object"]
    if isinstance(data_object, dict):
        for bb_key in data_object:
            if bb_key == "bndbox":
                print("bndbox: ", data_object[bb_key])
    else:
        for object_instance in data_object:
            print(
                "object name: ",
                object_instance["name"],
                ", bndbox: ",
                object_instance["bndbox"],
            )
