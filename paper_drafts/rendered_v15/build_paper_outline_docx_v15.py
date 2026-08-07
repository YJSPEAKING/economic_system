from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from lxml import etree


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "paper_drafts"
    / "rendered_v14"
    / "基于模仿学习的复杂经济系统主体决策模型小论文大纲稿_修改版v14.docx"
)
OUTPUT = (
    Path(__file__).resolve().parent
    / "基于模仿学习的复杂经济系统主体决策模型小论文大纲稿_修改版v15.docx"
)
MML2OMML_XSL = Path(
    r"C:\Program Files\Microsoft Office\Root\Office16\MML2OMML.XSL"
)

EXPECTED_SOURCE_SHA256 = (
    "CB07B0F5555576ACC97A7999B287402CC8A9ECD86ED727E480532755FA6FE6A4"
)

PARAGRAPH_REPLACEMENTS = {
    "为考察模仿学习信号与历史状态表征对复杂经济系统主体训练过程的影响": (
        "本节实验从模仿学习信号与历史状态表征两个层面，比较不同主体决策模型在复杂经济系统仿真中的训练表现。"
        "为此，本文选取TD3、GAIL+TD3和Transformer+GAIL+TD3三种主体决策模型进行对比。"
        "TD3作为基线模型，用于形成生产企业主体未引入模仿学习信号时的对照结果；"
        "GAIL+TD3在生产企业主体的TD3训练过程中引入生成对抗模仿学习信号，用于考察专家示范对训练过程的影响；"
        "Transformer+GAIL+TD3在GAIL+TD3基础上加入Transformer编码器，用于考察历史状态序列表征对训练过程的影响。"
        "其中，GAIL+TD3和Transformer+GAIL+TD3使用第五章整理得到的同一组专家数据，"
        "三种模型采用相同的仿真环境、训练进度尺度和统计方法。"
    ),
    "神经网络参数初始化、动作探索、训练样本抽取和经验池采样等过程均包含随机因素": (
        "为统一三种模型的训练进度与评价口径，本文以回合作为训练进度单位，"
        "并将每个评估时刻所包含的回合数记为E，取E=100。"
        "模型每完成E个连续训练回合，本文汇总一次评价指标，并将该汇总位置记为一个评估时刻（evaluation step）。"
        "第k个评估时刻对应第k组E个连续训练回合的统计结果。"
        "每个评估时刻均包含E个回合，单个回合的最大运行时长为100天。"
    ),
    "本文每完成100个训练回合汇总一次评价指标": (
        "本文在各评估时刻跟踪平均存活天数、生产企业累计收益、消费企业累计收益和银行累计收益。"
        "生产企业偿债能力用于训练后稳定运行阶段的风险评价。各项评价指标的定义如下。"
    ),
    "平均存活天数反映一个评估时刻内系统维持运行的平均时长": (
        "平均存活天数用于衡量系统在一个评估时刻内维持运行的平均时长。"
        "该指标等于第k个评估时刻所含E个回合存活天数的算术平均值，其定义如下："
    ),
    "累计收益分别包括生产企业累计收益、消费企业累计收益和银行累计收益": (
        "式中，L表示第k个评估时刻内第n个回合的存活天数。"
        "累计收益包括生产企业累计收益、消费企业累计收益和银行累计收益。"
        "对于企业主体，本文首先在单个回合内汇总收入、支出与利息，"
        "再对第k个评估时刻内E个回合的企业累计收益取算术平均值，其定义如下："
    ),
    "式中，J表示企业累计收益": (
        "式中，J表示企业在对应评估时刻的平均累计收益，R、C和I分别表示单个回合内的累计收入、累计支出和累计利息，"
        "i、k和n分别表示企业编号、评估时刻编号和该评估时刻内的回合编号。"
    ),
    "生产企业和消费企业分别依据上述定义计算累计收益": (
        "生产企业和消费企业分别依据上述定义计算平均累计收益。"
        "银行累计收益按照企业向银行支付的利息之和计算，"
        "并对第k个评估时刻内E个回合的结果取算术平均值，其定义如下："
    ),
    "式中，J表示银行累计收益": (
        "式中，J表示银行在对应评估时刻的平均累计收益，m表示系统中的企业主体数量，"
        "I表示对应企业在单个回合内支付的累计利息，k和n分别表示评估时刻编号和该评估时刻内的回合编号。"
    ),
    "每种方法在8组随机种子下形成8条指标序列": (
        "神经网络参数初始化、动作探索、训练样本抽取和经验池采样等过程均包含随机因素。"
        "本文将用于控制上述随机因素的联合配置统称为“随机种子”。"
        "三种模型分别在8组随机种子下独立运行，每种模型据此形成8条指标序列。"
        "本文在每个评估时刻计算8次独立运行结果的算术平均值，并以该均值作为曲线值。"
        "均值标准误（SEM: Standard Error of the Mean）用于描述8次独立运行均值的不确定性，其定义为："
    ),
}

SURVIVAL_MATHML = (
    '<math xmlns="http://www.w3.org/1998/Math/MathML">'
    '<mover><msub><mi>L</mi><mi>k</mi></msub><mo>¯</mo></mover><mo>=</mo>'
    '<mfrac><mn>1</mn><mi>E</mi></mfrac>'
    '<munderover><mo>∑</mo><mrow><mi>n</mi><mo>=</mo><mn>1</mn></mrow><mi>E</mi></munderover>'
    '<msub><mi>L</mi><mrow><mi>k</mi><mo>,</mo><mi>n</mi></mrow></msub>'
    '</math>'
)

ENTERPRISE_INCOME_MATHML = (
    '<math xmlns="http://www.w3.org/1998/Math/MathML">'
    '<mover><msubsup><mi>J</mi><mrow><mi>i</mi><mo>,</mo><mi>k</mi></mrow><mtext>firm</mtext></msubsup><mo>¯</mo></mover>'
    '<mo>=</mo><mfrac><mn>1</mn><mi>E</mi></mfrac>'
    '<munderover><mo>∑</mo><mrow><mi>n</mi><mo>=</mo><mn>1</mn></mrow><mi>E</mi></munderover>'
    '<mfenced><mrow>'
    '<mn>2</mn><msub><mi>R</mi><mrow><mi>i</mi><mo>,</mo><mi>k</mi><mo>,</mo><mi>n</mi></mrow></msub>'
    '<mo>−</mo><msub><mi>C</mi><mrow><mi>i</mi><mo>,</mo><mi>k</mi><mo>,</mo><mi>n</mi></mrow></msub>'
    '<mo>−</mo><msub><mi>I</mi><mrow><mi>i</mi><mo>,</mo><mi>k</mi><mo>,</mo><mi>n</mi></mrow></msub>'
    '</mrow></mfenced></math>'
)

BANK_INCOME_MATHML = (
    '<math xmlns="http://www.w3.org/1998/Math/MathML">'
    '<mover><msubsup><mi>J</mi><mi>k</mi><mtext>bank</mtext></msubsup><mo>¯</mo></mover>'
    '<mo>=</mo><mfrac><mn>1</mn><mi>E</mi></mfrac>'
    '<munderover><mo>∑</mo><mrow><mi>n</mi><mo>=</mo><mn>1</mn></mrow><mi>E</mi></munderover>'
    '<munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover>'
    '<msub><mi>I</mi><mrow><mi>i</mi><mo>,</mo><mi>k</mi><mo>,</mo><mi>n</mi></mrow></msub>'
    '</math>'
)


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest().upper()


def find_paragraph(document: Document, prefix: str):
    matches = [
        paragraph
        for paragraph in document.paragraphs
        if paragraph.text.startswith(prefix)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one paragraph beginning with {prefix!r}, found {len(matches)}."
        )
    return matches[0]


def replace_paragraph_text(paragraph, text: str) -> None:
    source_rpr = None
    for run in paragraph.runs:
        if run._r.rPr is not None:
            source_rpr = deepcopy(run._r.rPr)
            break

    for child in list(paragraph._p):
        if child.tag != qn("w:pPr"):
            paragraph._p.remove(child)

    run = paragraph.add_run(text)
    if source_rpr is not None:
        run._r.insert(0, source_rpr)


def mathml_to_omml(mathml: str):
    transform = etree.XSLT(etree.parse(str(MML2OMML_XSL)))
    result = transform(etree.fromstring(mathml.encode("utf-8")))
    return etree.fromstring(etree.tostring(result.getroot()))


def replace_equation_after(document: Document, paragraph_prefix: str, mathml: str) -> None:
    reference = find_paragraph(document, paragraph_prefix)
    paragraphs = document.paragraphs
    index = next(
        i for i, paragraph in enumerate(paragraphs) if paragraph._p is reference._p
    )
    equation_paragraph = paragraphs[index + 1]
    if not equation_paragraph._p.xpath(".//m:oMath"):
        raise RuntimeError(f"No Word equation found after {paragraph_prefix!r}.")

    for child in list(equation_paragraph._p):
        if child.tag != qn("w:pPr"):
            equation_paragraph._p.remove(child)
    equation_paragraph._p.append(mathml_to_omml(mathml))


def replace_cell_text(cell, text: str) -> None:
    paragraph = cell.paragraphs[0]
    replace_paragraph_text(paragraph, text)
    for extra_paragraph in list(cell.paragraphs[1:]):
        extra_paragraph._element.getparent().remove(extra_paragraph._element)


def update_table_6_1(document: Document) -> None:
    matching_tables = [
        table
        for table in document.tables
        if table.rows
        and [cell.text.strip() for cell in table.rows[0].cells] == ["项目", "设定"]
        and any(row.cells[0].text.strip() == "对比方法" for row in table.rows[1:])
    ]
    if len(matching_tables) != 1:
        raise RuntimeError(f"Expected one Table 6-1 candidate, found {len(matching_tables)}.")

    table = matching_tables[0]
    for row in table.rows[1:]:
        label = row.cells[0].text.strip()
        if label == "对比方法":
            replace_cell_text(row.cells[0], "对比模型")
        elif label == "随机种子组数":
            replace_cell_text(row.cells[1], "每种模型8组")
        elif label == "每个评估时刻对应回合数":
            replace_cell_text(row.cells[1], "E=100个连续训练回合")


def build() -> None:
    for required_path in (SOURCE, MML2OMML_XSL):
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    source_hash = file_sha256(SOURCE)
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            "The v14 source DOCX changed after the v15 builder was prepared. "
            f"Expected {EXPECTED_SOURCE_SHA256}, got {source_hash}."
        )

    document = Document(SOURCE)
    for prefix, replacement in PARAGRAPH_REPLACEMENTS.items():
        replace_paragraph_text(find_paragraph(document, prefix), replacement)

    replace_equation_after(
        document,
        "平均存活天数用于衡量系统在一个评估时刻内维持运行的平均时长",
        SURVIVAL_MATHML,
    )
    replace_equation_after(
        document,
        "式中，L表示第k个评估时刻内第n个回合的存活天数",
        ENTERPRISE_INCOME_MATHML,
    )
    replace_equation_after(
        document,
        "生产企业和消费企业分别依据上述定义计算平均累计收益",
        BANK_INCOME_MATHML,
    )
    update_table_6_1(document)
    document.save(OUTPUT)

    if file_sha256(SOURCE) != source_hash:
        raise RuntimeError("The v14 source DOCX was modified unexpectedly.")

    check = Document(OUTPUT)
    expected_fragments = [
        "本文选取TD3、GAIL+TD3和Transformer+GAIL+TD3三种主体决策模型进行对比",
        "每个评估时刻所包含的回合数记为E，取E=100",
        "本文将用于控制上述随机因素的联合配置统称为“随机种子”",
    ]
    all_text = "\n".join(paragraph.text for paragraph in check.paragraphs)
    for fragment in expected_fragments:
        if fragment not in all_text:
            raise RuntimeError(f"Missing expected text: {fragment!r}.")

    table = next(
        table
        for table in check.tables
        if table.rows and table.cell(0, 0).text.strip() == "项目"
    )
    rows = {row.cells[0].text.strip(): row.cells[1].text.strip() for row in table.rows[1:]}
    if rows.get("对比模型") != "TD3、GAIL+TD3、Transformer+GAIL+TD3":
        raise RuntimeError("Table 6-1 comparison-model row verification failed.")
    if rows.get("随机种子组数") != "每种模型8组":
        raise RuntimeError("Table 6-1 random-seed row verification failed.")
    if rows.get("每个评估时刻对应回合数") != "E=100个连续训练回合":
        raise RuntimeError("Table 6-1 evaluation-step row verification failed.")

    print(OUTPUT.resolve())


if __name__ == "__main__":
    build()
