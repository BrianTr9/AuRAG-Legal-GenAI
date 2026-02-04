#!/usr/bin/python3.12
# coding: utf-8
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom

def build_article_xml(article_hash: dict, article_caption_hash: dict, article_keys: list, out_path: str) -> None:
    # すべての num を集める（片方にしか無いキーも拾う）
    keys = set(article_hash.keys()) | set(article_caption_hash.keys())


    root = ET.Element("Articles")

    for num in article_keys:
        art_el = ET.SubElement(root, "Article")
        art_el.set("num", str(num))

        caption_el = ET.SubElement(art_el, "caption")
        caption_el.text = article_caption_hash.get(num, "")  # 無ければ空

        text_el = ET.SubElement(art_el, "text")
        text_el.text = article_hash.get(num, "")  # 無ければ空

    # きれいに整形して保存（ElementTree単体だと整形が弱いので minidom を使用）
    rough = ET.tostring(root, encoding="utf-8")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ", encoding="utf-8")

    with open(out_path, "wb") as f:
        f.write(pretty)
        
def remove_rt_in_ruby(elem: ET.Element) -> None:
    """
    elem 配下の <Ruby> の中にある <Rt> を削除する（読みを落としてベース文字だけ残す）。
    例: <Ruby>祀<Rt>し</Rt></Ruby> -> <Ruby>祀</Ruby>
    """
    for ruby in elem.findall(".//Ruby"):
        # 直下の Rt を削除（必要なら .//Rt にしてもOKだが通常は直下で十分）
        for rt in ruby.findall("Rt"):
            ruby.remove(rt)


def sentence_text(sentence_elem: ET.Element) -> str:
    """
    Sentence 要素から、Rt を除去した上で、タグを除去したテキストを返す。
    """
    remove_rt_in_ruby(sentence_elem)
    return "".join(sentence_elem.itertext())


def get_text(elem: ET.Element | None, default: str = "") -> str:
    if elem is None or elem.text is None:
        return default
    return elem.text


def extract_articles(article_xml_path: str):
    tree = ET.parse(article_xml_path)
    root = tree.getroot()

    article_hash = {}
    article_caption_hash = {}
    article_keys =[]
        
    # Ruby: doc.elements.each('//MainProvision//Article')
    for node in root.findall(".//MainProvision//Article"):
        num = node.get("Num")
        if num == "725":
            break
        num = num.replace("_", "-")
        article_keys.append(num)
        text = ""

        # ArticleTitle
        article_title = node.find("ArticleTitle")
        if article_title is not None and article_title.text is not None:
            text += sentence_text(article_title) + "　"  # 全角スペース

        # Paragraph
        for node2 in node.findall("Paragraph"):
            # ParagraphNum
            pnum_text = get_text(node2.find("ParagraphNum"), "")
            if pnum_text:
                text += pnum_text + "　"

            # ParagraphSentence/Sentence
            for s in node2.findall("ParagraphSentence/Sentence"):
                text += sentence_text(s)

            text += "\n"

            # Item
            for inode in node2.findall("Item"):
                text += get_text(inode.find("ItemTitle"), "") + "　"
                columns = []
                for cnode in inode.findall("./ItemSentence/Column"):
                    for cs in cnode.findall("./Sentence"):
                        columns.append(sentence_text(cs))
                text += "　".join(columns)

                items = []
                # ItemSentence//Sentence
                for s in inode.findall(".//ItemSentence/Sentence"):
                    items.append(sentence_text(s))

                text += "".join(items) + "\n"
                # SubItem
                for sinode in inode.findall(".//Subitem1"):
                    text += get_text(sinode.find("Subitem1Title"), "") + "　"
                    columns = []
                    # SubitemSentence/Column
                    for cnode in sinode.findall(".//Subitem1Sentence/Column"):
                        for cs in cnode.findall("./Sentence"):
                            columns.append(sentence_text(cs))
                    text += "　".join(columns)
                    subitems = []
                    # SubitemSentence/Sentence
                    for ssnode in sinode.findall(".//Subitem1Sentence/Sentence"):
                        subitems.append(sentence_text(ssnode))
                    text += "".join(subitems) + "\n"

        article_hash[num] = text.rstrip("\n")

        # ArticleCaption を付与した版
        caption = node.find("ArticleCaption")
        if caption is not None and caption.text is not None:
            caption = sentence_text(caption)

        article_caption_hash[num] = caption

    return article_hash, article_caption_hash, article_keys


def main():
    if len(sys.argv) < 3:
        print("Usage: python extract_text.py <articleXML> <outputXML>", file=sys.stderr)
        sys.exit(1)
    article_xml = sys.argv[1]
    out_xml = sys.argv[2]
    article_hash, article_caption_hash, article_keys = extract_articles(article_xml)
    build_article_xml(article_hash, article_caption_hash, article_keys, out_xml)

    # 動作確認の例：先頭数件を表示（不要なら削除してください）
    # for k in list(article_hash.keys())[:3]:
    #     print("====", k, "====")
    #     print(article_hash[k])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
