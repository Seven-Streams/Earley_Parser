from grammar_parser.pegVisitor import pegVisitor
from grammar_parser.pegLexer import pegLexer
from grammar_parser.pegParser import pegParser
from antlr4 import  CommonTokenStream, FileStream
def parse_text_with_visitor(file_path):
    # 创建输入流
    input_stream = FileStream(file_path)
    print(input_stream)
    # 创建词法分析器
    lexer = pegLexer(input_stream)
    
    # 创建词法符号流
    token_stream = CommonTokenStream(lexer)
    
    # 创建语法分析器
    parser = pegParser(token_stream)
    
    # 获取解析树（从起始规则开始）
    tree = parser.grammars()  # 替换为你的起始规则名
    print(tree)
    # 创建访问者
    visitor = pegVisitor()
    
    # 使用访问者访问解析树并获取结果
    result = visitor.visit(tree)
    
    return result, tree

file_path = "./python_grammar.txt"
result, parse_tree = parse_text_with_visitor(file_path)
print(result)