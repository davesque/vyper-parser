from typing import (
    Any,
    Sequence,
    Type,
    Union,
)

constant = Any
identifier = str
singleton = Union[None, bool]

AliasSeq = Sequence['alias']
ArgSeq = Sequence['arg']
CmpOpSeq = Sequence[Type['cmpop']]
ComprehensionSeq = Sequence['comprehension']
ExceptHandlerSeq = Sequence['excepthandler']
ExprSeq = Sequence['expr']
IdentifierSeq = Sequence['identifier']
KeywordSeq = Sequence['keyword']
SliceSeq = Sequence['slice']
StmtSeq = Sequence['stmt']
WithItemSeq = Sequence['withitem']


class VyperAST:
    __slots__ = ()


class mod(VyperAST):
    __slots__ = ()


class Module(mod):
    __slots__ = ('body',)

    def __init__(self,
                 body: StmtSeq):
        self.body = body


class Interactive(mod):
    __slots__ = ('body',)

    def __init__(self,
                 body: StmtSeq):
        self.body = body


class Expression(mod):
    __slots__ = ('body',)

    def __init__(self,
                 body: 'expr'):
        self.body = body


class PosAttributes:
    __slots__ = ('lineno', 'col_offset')

    def __init__(self,
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.lineno = lineno
        self.col_offset = col_offset


class stmt(PosAttributes, VyperAST):
    __slots__ = ()


class FunctionDef(stmt):
    __slots__ = ('name', 'args', 'body', 'decorator_list', 'returns')

    def __init__(self,
                 name: identifier,
                 args: 'arguments',
                 body: StmtSeq,
                 decorator_list: ExprSeq,
                 returns: 'expr' = None):
        self.name = name
        self.args = args
        self.body = body
        self.decorator_list = decorator_list
        self.returns = returns


class AsyncFunctionDef(stmt):
    __slots__ = ('name', 'args', 'body', 'decorator_list', 'returns')

    def __init__(self,
                 name: identifier,
                 args: 'arguments',
                 body: StmtSeq,
                 decorator_list: ExprSeq,
                 returns: 'expr' = None):
        self.name = name
        self.args = args
        self.body = body
        self.decorator_list = decorator_list
        self.returns = returns


class ClassDef(stmt):
    __slots__ = ('name', 'bases', 'keywords', 'body', 'decorator_list')

    def __init__(self,
                 name: identifier,
                 bases: ExprSeq,
                 keywords: KeywordSeq,
                 body: StmtSeq,
                 decorator_list: ExprSeq):
        self.name = name
        self.bases = bases
        self.keywords = keywords
        self.body = body
        self.decorator_list = decorator_list


class Return(stmt):
    __slots__ = ('value',)

    def __init__(self,
                 value: 'expr' = None):
        self.value = value


class Delete(stmt):
    __slots__ = ('targets',)

    def __init__(self,
                 targets: ExprSeq):
        self.targets = targets


class Assign(stmt):
    __slots__ = ('targets', 'value')

    def __init__(self,
                 targets: ExprSeq,
                 value: 'expr'):
        self.targets = targets
        self.value = value


class AugAssign(stmt):
    __slots__ = ('target', 'op', 'value')

    def __init__(self,
                 target: 'expr',
                 op: Type['operator'],
                 value: 'expr'):
        self.target = target
        self.op = op
        self.value = value


class AnnAssign(stmt):
    __slots__ = ('target', 'annotation', 'simple', 'value')

    def __init__(self,
                 target: 'expr',
                 annotation: 'expr',
                 simple: bool,
                 value: 'expr' = None):
        self.target = target
        self.annotation = annotation
        self.simple = simple
        self.value = value


class For(stmt):
    __slots__ = ('target', 'iter', 'body', 'orelse')

    def __init__(self,
                 target: 'expr',
                 iter: 'expr',
                 body: StmtSeq,
                 orelse: StmtSeq):
        self.target = target
        self.iter = iter
        self.body = body
        self.orelse = orelse


class AsyncFor(stmt):
    __slots__ = ('target', 'iter', 'body', 'orelse')

    def __init__(self,
                 target: 'expr',
                 iter: 'expr',
                 body: StmtSeq,
                 orelse: StmtSeq):
        self.target = target
        self.iter = iter
        self.body = body
        self.orelse = orelse


class While(stmt):
    __slots__ = ('test', 'body', 'orelse')

    def __init__(self,
                 test: 'expr',
                 body: StmtSeq,
                 orelse: StmtSeq):
        self.test = test
        self.body = body
        self.orelse = orelse


class If(stmt):
    __slots__ = ('test', 'body', 'orelse')

    def __init__(self,
                 test: 'expr',
                 body: StmtSeq,
                 orelse: StmtSeq):
        self.test = test
        self.body = body
        self.orelse = orelse


class With(stmt):
    __slots__ = ('items', 'body')

    def __init__(self,
                 items: WithItemSeq,
                 body: StmtSeq):
        self.items = items
        self.body = body


class AsyncWith(stmt):
    __slots__ = ('items', 'body')

    def __init__(self,
                 items: WithItemSeq,
                 body: StmtSeq):
        self.items = items
        self.body = body


class Raise(stmt):
    __slots__ = ('exc', 'cause')

    def __init__(self,
                 exc: 'expr' = None,
                 cause: 'expr' = None):
        self.exc = exc
        self.cause = cause


class Try(stmt):
    __slots__ = ('body', 'handlers', 'orelse', 'finalbody')

    def __init__(self,
                 body: StmtSeq,
                 handlers: ExceptHandlerSeq,
                 orelse: StmtSeq,
                 finalbody: StmtSeq):
        self.body = body
        self.handlers = handlers
        self.orelse = orelse
        self.finalbody = finalbody


class Assert(stmt):
    __slots__ = ('test', 'msg')

    def __init__(self,
                 test: 'expr',
                 msg: 'expr' = None):
        self.test = test
        self.msg = msg


class Import(stmt):
    __slots__ = ('names',)

    def __init__(self,
                 names: AliasSeq):
        self.names = names


class ImportFrom(stmt):
    __slots__ = ('names', 'module', 'level')

    def __init__(self,
                 names: AliasSeq,
                 module: identifier = None,
                 level: int = None):
        self.names = names
        self.module = module
        self.level = level


class Global(stmt):
    __slots__ = ('names',)

    def __init__(self,
                 names: IdentifierSeq):
        self.names = names


class Nonlocal(stmt):
    __slots__ = ('names',)

    def __init__(self,
                 names: IdentifierSeq):
        self.names = names


class Expr(stmt):
    __slots__ = ('value',)

    def __init__(self,
                 value: 'expr'):
        self.value = value


class Pass(stmt):
    __slots__ = ()


class Break(stmt):
    __slots__ = ()


class Continue(stmt):
    __slots__ = ()


class expr(PosAttributes, VyperAST):
    __slots__ = ()


class BoolOp(expr):
    __slots__ = ('op', 'values')

    def __init__(self,
                 op: Type['boolop'],
                 values: ExprSeq):
        self.op = op
        self.values = values


class BinOp(expr):
    __slots__ = ('left', 'op', 'right')

    def __init__(self,
                 left: expr,
                 op: Type['operator'],
                 right: expr):
        self.left = left
        self.op = op
        self.right = right


class UnaryOp(expr):
    __slots__ = ('op', 'operand')

    def __init__(self,
                 op: Type['unaryop'],
                 operand: expr):
        self.op = op
        self.operand = operand


class Lambda(expr):
    __slots__ = ('args', 'body')

    def __init__(self,
                 args: 'arguments',
                 body: expr):
        self.args = args
        self.body = body


class IfExp(expr):
    __slots__ = ('test', 'body', 'orelse')

    def __init__(self,
                 test: expr,
                 body: expr,
                 orelse: expr):
        self.test = test
        self.body = body
        self.orelse = orelse


class Dict(expr):
    __slots__ = ('keys', 'values')

    def __init__(self,
                 keys: ExprSeq,
                 values: ExprSeq):
        self.keys = keys
        self.values = values


class Set(expr):
    __slots__ = ('elts',)

    def __init__(self,
                 elts: ExprSeq):
        self.elts = elts


class ListComp(expr):
    __slots__ = ('elt', 'generators')

    def __init__(self,
                 elt: expr,
                 generators: ComprehensionSeq):
        self.elt = elt
        self.generators = generators


class SetComp(expr):
    __slots__ = ('elt', 'generators')

    def __init__(self,
                 elt: expr,
                 generators: ComprehensionSeq):
        self.elt = elt
        self.generators = generators


class DictComp(expr):
    __slots__ = ('key', 'value', 'generators')

    def __init__(self,
                 key: expr,
                 value: expr,
                 generators: ComprehensionSeq):
        self.key = key
        self.value = value
        self.generators = generators


class GeneratorExp(expr):
    __slots__ = ('elt', 'generators')

    def __init__(self,
                 elt: expr,
                 generators: ComprehensionSeq):
        self.elt = elt
        self.generators = generators


class Await(expr):
    __slots__ = ('value',)

    def __init__(self,
                 value: expr):
        self.value = value


class Yield(expr):
    __slots__ = ('value',)

    def __init__(self,
                 value: expr = None):
        self.value = value


class YieldFrom(expr):
    __slots__ = ('value',)

    def __init__(self,
                 value: expr):
        self.value = value


class Compare(expr):
    __slots__ = ('left', 'ops', 'comparators')

    def __init__(self,
                 left: expr,
                 ops: CmpOpSeq,
                 comparators: ExprSeq):
        self.left = left
        self.ops = ops
        self.comparators = comparators


class Call(expr):
    __slots__ = ('func', 'args', 'keywords')

    def __init__(self,
                 func: expr,
                 args: ExprSeq,
                 keywords: KeywordSeq):
        self.func = func
        self.args = args
        self.keywords = keywords


class Num(expr):
    __slots__ = ('n',)

    def __init__(self,
                 n: Union[int, float]):
        self.n = n


class Str(expr):
    __slots__ = ('s',)

    def __init__(self,
                 s: str):
        self.s = s


class FormattedValue(expr):
    __slots__ = ('value', 'conversion', 'format_spec')

    def __init__(self,
                 value: expr,
                 conversion: int = None,
                 format_spec: expr = None):
        self.value = value
        self.conversion = conversion
        self.format_spec = format_spec


class JoinedStr(expr):
    __slots__ = ('values',)

    def __init__(self,
                 values: ExprSeq):
        self.values = values


class Bytes(expr):
    __slots__ = ('s',)

    def __init__(self,
                 s: bytes):
        self.s = s


class NameConstant(expr):
    __slots__ = ('value',)

    def __init__(self,
                 value: singleton):
        self.value = value


class Ellipsis(expr):
    __slots__ = ()


class Constant(expr):
    __slots__ = ('value',)

    def __init__(self,
                 value: constant):
        self.value = value


class Attribute(expr):
    __slots__ = ('value', 'attr', 'ctx')

    def __init__(self,
                 value: expr,
                 attr: identifier,
                 ctx: 'expr_context'):
        self.value = value
        self.attr = attr
        self.ctx = ctx


class Subscript(expr):
    __slots__ = ('value', 'slice', 'ctx')

    def __init__(self,
                 value: expr,
                 slice: slice,
                 ctx: 'expr_context'):
        self.value = value
        self.slice = slice
        self.ctx = ctx


class Starred(expr):
    __slots__ = ('value', 'ctx')

    def __init__(self,
                 value: expr,
                 ctx: 'expr_context'):
        self.value = value
        self.ctx = ctx


class Name(expr):
    __slots__ = ('id', 'ctx')

    def __init__(self,
                 id: identifier,
                 ctx: 'expr_context'):
        self.id = id
        self.ctx = ctx


class List(expr):
    __slots__ = ('elts', 'ctx')

    def __init__(self,
                 elts: ExprSeq,
                 ctx: 'expr_context'):
        self.elts = elts
        self.ctx = ctx


class Tuple(expr):
    __slots__ = ('elts', 'ctx')

    def __init__(self,
                 elts: ExprSeq,
                 ctx: 'expr_context'):
        self.elts = elts
        self.ctx = ctx


class expr_context(VyperAST):
    pass


class Load(expr_context):
    pass


class Store(expr_context):
    pass


class Del(expr_context):
    pass


class AugLoad(expr_context):
    pass


class AugStore(expr_context):
    pass


class Param(expr_context):
    pass


class slice(VyperAST):
    __slots__ = ()


class Slice(slice):
    __slots__ = ('lower', 'upper', 'step')

    def __init__(self,
                 lower: 'expr' = None,
                 upper: 'expr' = None,
                 step: 'expr' = None):
        self.lower = lower
        self.upper = upper
        self.step = step


class ExtSlice(slice):
    __slots__ = ('dims',)

    def __init__(self,
                 dims: SliceSeq):
        self.dims = dims


class Index(slice):
    __slots__ = ('value',)

    def __init__(self,
                 value: 'expr'):
        self.value = value


class boolop(VyperAST):
    pass


class And(boolop):
    pass


class Or(boolop):
    pass


class operator(VyperAST):
    pass


class Add(operator):
    pass


class Sub(operator):
    pass


class Mult(operator):
    pass


class MatMult(operator):
    pass


class Div(operator):
    pass


class Mod(operator):
    pass


class Pow(operator):
    pass


class LShift(operator):
    pass


class RShift(operator):
    pass


class BitOr(operator):
    pass


class BitXor(operator):
    pass


class BitAnd(operator):
    pass


class FloorDiv(operator):
    pass


class unaryop(VyperAST):
    pass


class Invert(unaryop):
    pass


class Not(unaryop):
    pass


class UAdd(unaryop):
    pass


class USub(unaryop):
    pass


class cmpop(VyperAST):
    pass


class Eq(cmpop):
    pass


class NotEq(cmpop):
    pass


class Lt(cmpop):
    pass


class LtE(cmpop):
    pass


class Gt(cmpop):
    pass


class GtE(cmpop):
    pass


class Is(cmpop):
    pass


class IsNot(cmpop):
    pass


class In(cmpop):
    pass


class NotIn(cmpop):
    pass


class comprehension(VyperAST):
    __slots__ = ('target', 'iter', 'ifs', 'is_async')

    def __init__(self,
                 target: expr,
                 iter: expr,
                 ifs: ExprSeq,
                 is_async: bool):
        self.target = target
        self.iter = iter
        self.ifs = ifs
        self.is_async = is_async


class excepthandler(PosAttributes, VyperAST):
    __slots__ = ()


class ExceptHandler(excepthandler):
    __slots__ = ('body', 'type', 'name')

    def __init__(self,
                 body: StmtSeq,
                 type: expr = None,
                 name: identifier = None):
        self.body = body
        self.type = type
        self.name = name


class arguments(VyperAST):
    __slots__ = ('args', 'kwonlyargs', 'kw_defaults', 'defaults', 'vararg', 'kwarg')

    def __init__(self,
                 args: ArgSeq,
                 kwonlyargs: ArgSeq,
                 kw_defaults: ExprSeq,
                 defaults: ExprSeq,
                 vararg: 'arg' = None,
                 kwarg: 'arg' = None):
        self.args = args
        self.kwonlyargs = kwonlyargs
        self.kw_defaults = kw_defaults
        self.defaults = defaults
        self.vararg = vararg
        self.kwarg = kwarg


class arg(PosAttributes, VyperAST):
    __slots__ = ('arg', 'annotation')

    def __init__(self,
                 arg: identifier,
                 annotation: expr = None):
        self.arg = arg
        self.annotation = annotation


class keyword(VyperAST):
    __slots__ = ('value', 'arg')

    def __init__(self,
                 value: expr,
                 arg: identifier = None):
        self.value = value
        self.arg = arg


class alias(VyperAST):
    __slots__ = ('name', 'asname')

    def __init__(self,
                 name: identifier,
                 asname: identifier = None):
        self.name = name
        self.asname = asname


class withitem(VyperAST):
    __slots__ = ('context_expr', 'optional_vars')

    def __init__(self,
                 context_expr: expr,
                 optional_vars: expr = None):
        self.context_expr = context_expr
        self.optional_vars = optional_vars
