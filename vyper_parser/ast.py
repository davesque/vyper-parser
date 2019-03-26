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
                 returns: 'expr' = None,
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.name = name
        self.args = args
        self.body = body
        self.decorator_list = decorator_list
        self.returns = returns

        super().__init__(lineno=lineno, col_offset=col_offset)


class AsyncFunctionDef(stmt):
    __slots__ = ('name', 'args', 'body', 'decorator_list', 'returns')

    def __init__(self,
                 name: identifier,
                 args: 'arguments',
                 body: StmtSeq,
                 decorator_list: ExprSeq,
                 returns: 'expr' = None,
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.name = name
        self.args = args
        self.body = body
        self.decorator_list = decorator_list
        self.returns = returns

        super().__init__(lineno=lineno, col_offset=col_offset)


class ClassDef(stmt):
    __slots__ = ('name', 'bases', 'keywords', 'body', 'decorator_list')

    def __init__(self,
                 name: identifier,
                 bases: ExprSeq,
                 keywords: KeywordSeq,
                 body: StmtSeq,
                 decorator_list: ExprSeq,
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.name = name
        self.bases = bases
        self.keywords = keywords
        self.body = body
        self.decorator_list = decorator_list

        super().__init__(lineno=lineno, col_offset=col_offset)


class Return(stmt):
    __slots__ = ('value',)

    def __init__(self,
                 value: 'expr' = None,
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.value = value

        super().__init__(lineno=lineno, col_offset=col_offset)


class Delete(stmt):
    __slots__ = ('targets',)

    def __init__(self,
                 targets: ExprSeq,
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.targets = targets

        super().__init__(lineno=lineno, col_offset=col_offset)


class Assign(stmt):
    __slots__ = ('targets', 'value')

    def __init__(self,
                 targets: ExprSeq,
                 value: 'expr',
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.targets = targets
        self.value = value

        super().__init__(lineno=lineno, col_offset=col_offset)


class AugAssign(stmt):
    __slots__ = ('target', 'op', 'value')

    def __init__(self,
                 target: 'expr',
                 op: Type['operator'],
                 value: 'expr',
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.target = target
        self.op = op
        self.value = value

        super().__init__(lineno=lineno, col_offset=col_offset)


class AnnAssign(stmt):
    __slots__ = ('target', 'annotation', 'simple', 'value')

    def __init__(self,
                 target: 'expr',
                 annotation: 'expr',
                 simple: bool,
                 value: 'expr' = None,
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.target = target
        self.annotation = annotation
        self.simple = simple
        self.value = value

        super().__init__(lineno=lineno, col_offset=col_offset)


class For(stmt):
    __slots__ = ('target', 'iter', 'body', 'orelse')

    def __init__(self,
                 target: 'expr',
                 iter: 'expr',
                 body: StmtSeq,
                 orelse: StmtSeq,
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.target = target
        self.iter = iter
        self.body = body
        self.orelse = orelse

        super().__init__(lineno=lineno, col_offset=col_offset)


class AsyncFor(stmt):
    __slots__ = ('target', 'iter', 'body', 'orelse')

    def __init__(self,
                 target: 'expr',
                 iter: 'expr',
                 body: StmtSeq,
                 orelse: StmtSeq,
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.target = target
        self.iter = iter
        self.body = body
        self.orelse = orelse

        super().__init__(lineno=lineno, col_offset=col_offset)


class While(stmt):
    __slots__ = ('test', 'body', 'orelse')

    def __init__(self,
                 test: 'expr',
                 body: StmtSeq,
                 orelse: StmtSeq,
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.test = test
        self.body = body
        self.orelse = orelse

        super().__init__(lineno=lineno, col_offset=col_offset)


class If(stmt):
    __slots__ = ('test', 'body', 'orelse')

    def __init__(self,
                 test: 'expr',
                 body: StmtSeq,
                 orelse: StmtSeq,
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.test = test
        self.body = body
        self.orelse = orelse

        super().__init__(lineno=lineno, col_offset=col_offset)


class With(stmt):
    __slots__ = ('items', 'body')

    def __init__(self,
                 items: WithItemSeq,
                 body: StmtSeq,
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.items = items
        self.body = body

        super().__init__(lineno=lineno, col_offset=col_offset)


class AsyncWith(stmt):
    __slots__ = ('items', 'body')

    def __init__(self,
                 items: WithItemSeq,
                 body: StmtSeq,
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.items = items
        self.body = body

        super().__init__(lineno=lineno, col_offset=col_offset)


class Raise(stmt):
    __slots__ = ('exc', 'cause')

    def __init__(self,
                 exc: 'expr' = None,
                 cause: 'expr' = None,
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.exc = exc
        self.cause = cause

        super().__init__(lineno=lineno, col_offset=col_offset)


class Try(stmt):
    __slots__ = ('body', 'handlers', 'orelse', 'finalbody')

    def __init__(self,
                 body: StmtSeq,
                 handlers: ExceptHandlerSeq,
                 orelse: StmtSeq,
                 finalbody: StmtSeq,
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.body = body
        self.handlers = handlers
        self.orelse = orelse
        self.finalbody = finalbody

        super().__init__(lineno=lineno, col_offset=col_offset)


class Assert(stmt):
    __slots__ = ('test', 'msg')

    def __init__(self,
                 test: 'expr',
                 msg: 'expr' = None,
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.test = test
        self.msg = msg

        super().__init__(lineno=lineno, col_offset=col_offset)


class Import(stmt):
    __slots__ = ('names',)

    def __init__(self,
                 names: AliasSeq,
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.names = names

        super().__init__(lineno=lineno, col_offset=col_offset)


class ImportFrom(stmt):
    __slots__ = ('names', 'module', 'level')

    def __init__(self,
                 names: AliasSeq,
                 module: identifier = None,
                 level: int = None,
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.names = names
        self.module = module
        self.level = level

        super().__init__(lineno=lineno, col_offset=col_offset)


class Global(stmt):
    __slots__ = ('names',)

    def __init__(self,
                 names: IdentifierSeq,
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.names = names

        super().__init__(lineno=lineno, col_offset=col_offset)


class Nonlocal(stmt):
    __slots__ = ('names',)

    def __init__(self,
                 names: IdentifierSeq,
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.names = names

        super().__init__(lineno=lineno, col_offset=col_offset)


class Expr(stmt):
    __slots__ = ('value',)

    def __init__(self,
                 value: 'expr',
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.value = value

        super().__init__(lineno=lineno, col_offset=col_offset)


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
                 values: ExprSeq,
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.op = op
        self.values = values

        super().__init__(lineno=lineno, col_offset=col_offset)


class BinOp(expr):
    __slots__ = ('left', 'op', 'right')

    def __init__(self,
                 left: expr,
                 op: Type['operator'],
                 right: expr,
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.left = left
        self.op = op
        self.right = right

        super().__init__(lineno=lineno, col_offset=col_offset)


class UnaryOp(expr):
    __slots__ = ('op', 'operand')

    def __init__(self,
                 op: Type['unaryop'],
                 operand: expr,
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.op = op
        self.operand = operand

        super().__init__(lineno=lineno, col_offset=col_offset)


class Lambda(expr):
    __slots__ = ('args', 'body')

    def __init__(self,
                 args: 'arguments',
                 body: expr,
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.args = args
        self.body = body

        super().__init__(lineno=lineno, col_offset=col_offset)


class IfExp(expr):
    __slots__ = ('test', 'body', 'orelse')

    def __init__(self,
                 test: expr,
                 body: expr,
                 orelse: expr,
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.test = test
        self.body = body
        self.orelse = orelse

        super().__init__(lineno=lineno, col_offset=col_offset)


class Dict(expr):
    __slots__ = ('keys', 'values')

    def __init__(self,
                 keys: ExprSeq,
                 values: ExprSeq,
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.keys = keys
        self.values = values

        super().__init__(lineno=lineno, col_offset=col_offset)


class Set(expr):
    __slots__ = ('elts',)

    def __init__(self,
                 elts: ExprSeq,
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.elts = elts

        super().__init__(lineno=lineno, col_offset=col_offset)


class ListComp(expr):
    __slots__ = ('elt', 'generators')

    def __init__(self,
                 elt: expr,
                 generators: ComprehensionSeq,
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.elt = elt
        self.generators = generators

        super().__init__(lineno=lineno, col_offset=col_offset)


class SetComp(expr):
    __slots__ = ('elt', 'generators')

    def __init__(self,
                 elt: expr,
                 generators: ComprehensionSeq,
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.elt = elt
        self.generators = generators

        super().__init__(lineno=lineno, col_offset=col_offset)


class DictComp(expr):
    __slots__ = ('key', 'value', 'generators')

    def __init__(self,
                 key: expr,
                 value: expr,
                 generators: ComprehensionSeq,
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.key = key
        self.value = value
        self.generators = generators

        super().__init__(lineno=lineno, col_offset=col_offset)


class GeneratorExp(expr):
    __slots__ = ('elt', 'generators')

    def __init__(self,
                 elt: expr,
                 generators: ComprehensionSeq,
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.elt = elt
        self.generators = generators

        super().__init__(lineno=lineno, col_offset=col_offset)


class Await(expr):
    __slots__ = ('value',)

    def __init__(self,
                 value: expr,
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.value = value

        super().__init__(lineno=lineno, col_offset=col_offset)


class Yield(expr):
    __slots__ = ('value',)

    def __init__(self,
                 value: expr = None,
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.value = value

        super().__init__(lineno=lineno, col_offset=col_offset)


class YieldFrom(expr):
    __slots__ = ('value',)

    def __init__(self,
                 value: expr,
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.value = value

        super().__init__(lineno=lineno, col_offset=col_offset)


class Compare(expr):
    __slots__ = ('left', 'ops', 'comparators')

    def __init__(self,
                 left: expr,
                 ops: CmpOpSeq,
                 comparators: ExprSeq,
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.left = left
        self.ops = ops
        self.comparators = comparators

        super().__init__(lineno=lineno, col_offset=col_offset)


class Call(expr):
    __slots__ = ('func', 'args', 'keywords')

    def __init__(self,
                 func: expr,
                 args: ExprSeq,
                 keywords: KeywordSeq,
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.func = func
        self.args = args
        self.keywords = keywords

        super().__init__(lineno=lineno, col_offset=col_offset)


class Num(expr):
    __slots__ = ('n',)

    def __init__(self,
                 n: Union[int, float],
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.n = n

        super().__init__(lineno=lineno, col_offset=col_offset)


class Str(expr):
    __slots__ = ('s',)

    def __init__(self,
                 s: str,
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.s = s

        super().__init__(lineno=lineno, col_offset=col_offset)


class FormattedValue(expr):
    __slots__ = ('value', 'conversion', 'format_spec')

    def __init__(self,
                 value: expr,
                 conversion: int = None,
                 format_spec: expr = None,
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.value = value
        self.conversion = conversion
        self.format_spec = format_spec

        super().__init__(lineno=lineno, col_offset=col_offset)


class JoinedStr(expr):
    __slots__ = ('values',)

    def __init__(self,
                 values: ExprSeq,
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.values = values

        super().__init__(lineno=lineno, col_offset=col_offset)


class Bytes(expr):
    __slots__ = ('s',)

    def __init__(self,
                 s: bytes,
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.s = s

        super().__init__(lineno=lineno, col_offset=col_offset)


class NameConstant(expr):
    __slots__ = ('value',)

    def __init__(self,
                 value: singleton,
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.value = value

        super().__init__(lineno=lineno, col_offset=col_offset)


class Ellipsis(expr):
    __slots__ = ()


class Constant(expr):
    __slots__ = ('value',)

    def __init__(self,
                 value: constant,
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.value = value

        super().__init__(lineno=lineno, col_offset=col_offset)


class Attribute(expr):
    __slots__ = ('value', 'attr', 'ctx')

    def __init__(self,
                 value: expr,
                 attr: identifier,
                 ctx: 'expr_context',
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.value = value
        self.attr = attr
        self.ctx = ctx

        super().__init__(lineno=lineno, col_offset=col_offset)


class Subscript(expr):
    __slots__ = ('value', 'slice', 'ctx')

    def __init__(self,
                 value: expr,
                 slice: slice,
                 ctx: 'expr_context',
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.value = value
        self.slice = slice
        self.ctx = ctx

        super().__init__(lineno=lineno, col_offset=col_offset)


class Starred(expr):
    __slots__ = ('value', 'ctx')

    def __init__(self,
                 value: expr,
                 ctx: 'expr_context',
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.value = value
        self.ctx = ctx

        super().__init__(lineno=lineno, col_offset=col_offset)


class Name(expr):
    __slots__ = ('id', 'ctx')

    def __init__(self,
                 id: identifier,
                 ctx: 'expr_context',
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.id = id
        self.ctx = ctx

        super().__init__(lineno=lineno, col_offset=col_offset)


class List(expr):
    __slots__ = ('elts', 'ctx')

    def __init__(self,
                 elts: ExprSeq,
                 ctx: 'expr_context',
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.elts = elts
        self.ctx = ctx

        super().__init__(lineno=lineno, col_offset=col_offset)


class Tuple(expr):
    __slots__ = ('elts', 'ctx')

    def __init__(self,
                 elts: ExprSeq,
                 ctx: 'expr_context',
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.elts = elts
        self.ctx = ctx

        super().__init__(lineno=lineno, col_offset=col_offset)


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
                 name: identifier = None,
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.body = body
        self.type = type
        self.name = name

        super().__init__(lineno=lineno, col_offset=col_offset)


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
                 annotation: expr = None,
                 *,
                 lineno: int = None,
                 col_offset: int = None):
        self.arg = arg
        self.annotation = annotation

        super().__init__(lineno=lineno, col_offset=col_offset)


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
