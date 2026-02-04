#!/usr/bin/python3.12
# coding: utf-8
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom

def load_articles_attr_xml(articles_xml_path: str) -> dict[str, str]:
    """
    事前に作った Articles XML を読み込んで辞書を作る。
    想定:
      <Articles>
        <Article num="15" caption="...." text="...." />
        <Article num="398-6" caption="...." text="...." />
      </Articles>

    戻り値:
      { "15": caption_or_text, "398-6": caption_or_text, ... }
    """
    tree = ET.parse(articles_xml_path)
    root = tree.getroot()

    article_map: dict[str, str] = {}
    for a in root.findall(".//Article"):
        num = a.get("num")
        if not num:
            continue

        # 優先: caption（caption+本文が入っている想定）
        caption_elem = a.find("caption")
        text_elem = a.find("text")

        caption = caption_elem.text if caption_elem is not None else ""
        text = text_elem.text if text_elem is not None else ""
        val = caption + "\n" + text if (caption is not None and caption != "") else (text or "")
        # 末尾の改行だけ落としておく（必要なら外してOK）
        val = val.rstrip("\n")
        article_map[num] = val

    return article_map


def prettify_xml(elem: ET.Element) -> bytes:
    rough = ET.tostring(elem, encoding="UTF-8")
    pretty = minidom.parseString(rough).toprettyxml(indent="", encoding="UTF-8")
    return b"\n".join(
        line for line in pretty.splitlines() if line.strip()
    )


def replace_articles_with_t1(article_map: dict[str, str], dataset_in: str, dataset_out: str) -> None:
    tree = ET.parse(dataset_in)
    root = tree.getroot()

    # dataset/pair を走査
    for pair in root.findall(".//pair"):
        # pair内のarticle要素を順番通り取得
        article_elems = pair.findall("article")
        if not article_elems:
            continue

        nums = [a.text or "" for a in article_elems if (a.text or "").strip()]

        # 条文本文を連結（元の順番）
        texts = []
        for n in nums:
            if n in article_map:
                texts.append(article_map[n])
            else:
                # 見つからない場合は空にせず、分かるようにマーカーを入れる（不要なら "" に）
                texts.append(f"[MISSING ARTICLE {n}]")

        t1_text = "\n".join(texts).rstrip("\n")

        # 既存の <article> を削除
        for a in article_elems:
            pair.remove(a)

        # <t1> を作って挿入（基本は t2 の前に入れる）
        t1_elem = ET.Element("t1")
        t1_elem.text = t1_text

        # t2 の位置を探す
        children = list(pair)
        t2_index = None
        for i, ch in enumerate(children):
            if ch.tag == "t2":
                t2_index = i
                break

        if t2_index is None:
            pair.append(t1_elem)
        else:
            pair.insert(t2_index, t1_elem)

    # ルートを整形して保存（xml宣言付き）
    pretty = prettify_xml(root)
    with open(dataset_out, "wb") as f:
        f.write(pretty)


def main():
    if len(sys.argv) < 4:
        print(
            "Usage: python makeEntailment.py <articles_attr.xml> <dataset_in.xml> <dataset_out.xml>",
            file=sys.stderr,
        )
        sys.exit(1)

    articles_xml = sys.argv[1]
    dataset_in = sys.argv[2]
    dataset_out = sys.argv[3]

    article_map = load_articles_attr_xml(articles_xml)
    replace_articles_with_t1(article_map, dataset_in, dataset_out)


if __name__ == "__main__":
    main()
