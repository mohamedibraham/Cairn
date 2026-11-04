import ast
from dataclasses import dataclass


MAX_PARSE_CHARS = 2_000_000


MODULE_LEVEL_KINDS = frozenset({"module_doc", "imports", "constants", "script"})


class PythonParseError(Exception):
    pass

@dataclass(frozen=True)
class CodeUnit:
    kind: str  # module_doc | imports | constants | script | class | function | method
    name: str  # qualified ("Folder.run") or a placeholder like "<imports>"
    start_line: int
    end_line: int
    text: str
    signature: str | None = None
    parent: str | None = None
    decorators: tuple = ()
    bases: tuple = ()
    docstring: str | None = None
    imports_used: tuple = ()
    imports_declared: tuple = ()  # only on "imports" units
    calls: tuple = ()
    is_async: bool = False


def parse_python(source: str) -> list:
    """Parses `source` into CodeUnits in file order. Raises PythonParseError."""
    if len(source) > MAX_PARSE_CHARS:
        raise PythonParseError("File is too large for structural parsing")

    
    source = source.replace("\r\n", "\n").replace("\r", "\n")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        where = f" (line {exc.lineno})" if exc.lineno else ""
        raise PythonParseError(f"Python syntax error{where}: {exc.msg}") from exc
    except (ValueError, RecursionError, MemoryError) as exc:
        raise PythonParseError(f"Could not parse Python source: {exc}") from exc

    return _Builder(source, tree).build()



def _is_docstring(node) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _start(node) -> int:
    
    decorators = getattr(node, "decorator_list", [])
    return min([node.lineno] + [d.lineno for d in decorators])


def _import_pairs(node) -> list:
    """(bound_name, qualified_name) for an Import / ImportFrom node."""
    pairs = []
    if isinstance(node, ast.Import):
        for alias in node.names:
            bound = alias.asname or alias.name.split(".")[0]
            pairs.append((bound, alias.name))
    elif isinstance(node, ast.ImportFrom):
        base = "." * node.level + (node.module or "")
        sep = "" if base.endswith(".") else "."
        for alias in node.names:
            if alias.name == "*":
                continue
            pairs.append((alias.asname or alias.name, base + sep + alias.name))
    return pairs


def _is_main_guard(node) -> bool:
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
    )


def _category(node) -> str:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return "imports"
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return "def"
    return "stmt"


def _dedupe(items) -> tuple:
    return tuple(dict.fromkeys(items))



class _Builder:
    def __init__(self, source: str, tree: ast.Module):
        self.lines = source.split("\n")
        self.tree = tree

        
        self.imports = {}
        for node in ast.walk(tree):
            for bound, qualified in _import_pairs(node):
                self.imports[bound] = qualified

        self.top_symbols = {
            n.name
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        self.class_methods = {}
        self._collect_class_methods(tree.body, None)

    

    def _collect_class_methods(self, body, parent):
        for node in body:
            if isinstance(node, ast.ClassDef):
                qname = f"{parent}.{node.name}" if parent else node.name
                self.class_methods[qname] = {
                    c.name
                    for c in node.body
                    if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                self._collect_class_methods(node.body, qname)

    

    def _with_leading_comments(self, start: int, floor: int) -> int:
        
        while start - 1 > floor and self.lines[start - 2].lstrip().startswith("#"):
            start -= 1
        return start

    def _slice(self, start: int, end: int) -> str:
        return "\n".join(self.lines[start - 1:end]).rstrip()


    def build(self) -> list:
        units = []
        floor = 0
        run = []
        run_category = None

        def flush():
            nonlocal floor, run, run_category
            if run:
                unit = self._run_unit(run, run_category, floor)
                units.append(unit)
                floor = unit.end_line
            run, run_category = [], None

        for index, node in enumerate(self.tree.body):
            if index == 0 and _is_docstring(node):
                start = self._with_leading_comments(node.lineno, floor)
                units.append(CodeUnit(
                    kind="module_doc", name="<module docstring>",
                    start_line=start, end_line=node.end_lineno,
                    text=self._slice(start, node.end_lineno),
                    docstring=ast.get_docstring(self.tree),
                ))
                floor = node.end_lineno
                continue

            if _is_main_guard(node):
                
                flush()
                run, run_category = [node], "stmt"
                flush()
                continue

            category = _category(node)
            if category == "def":
                flush()
                if isinstance(node, ast.ClassDef):
                    new_units = self._class_units(node, None, floor)
                else:
                    new_units = [self._function_unit(node, None, floor, None)]
                units.extend(new_units)
                floor = node.end_lineno
            else:
                if run and category != run_category:
                    flush()
                run.append(node)
                run_category = category
        flush()
        return units

    def _run_unit(self, nodes: list, category: str, floor: int) -> CodeUnit:
        start = self._with_leading_comments(_start(nodes[0]), floor)
        end = nodes[-1].end_lineno
        text = self._slice(start, end)

        if category == "imports":
            declared = _dedupe(q for n in nodes for _, q in _import_pairs(n))
            return CodeUnit(
                kind="imports", name="<imports>", start_line=start, end_line=end,
                text=text, imports_declared=declared,
            )

        if len(nodes) == 1 and _is_main_guard(nodes[0]):
            kind, name = "script", "__main__"
        elif all(isinstance(n, (ast.Assign, ast.AnnAssign)) for n in nodes):
            kind, name = "constants", "<constants>"
        else:
            kind, name = "script", "<module code>"

        wrapper = ast.Module(body=nodes, type_ignores=[])
        return CodeUnit(
            kind=kind, name=name, start_line=start, end_line=end, text=text,
            imports_used=self._imports_used(wrapper),
            calls=self._calls(wrapper, None, name),
        )


    def _class_units(self, node: ast.ClassDef, parent, floor: int) -> list:
        qname = f"{parent}.{node.name}" if parent else node.name
        start = self._with_leading_comments(_start(node), floor)
        end = node.end_lineno
        body = node.body

        has_doc = bool(body) and _is_docstring(body[0])
        if has_doc:
            header_end = body[0].end_lineno
        else:
            header_end = _start(body[0]) - 1 if body else node.lineno
        header_end = max(header_end, node.lineno)

        parts = [self._slice(start, header_end)]

        
        members = body[1:] if has_doc else body
        for child in members:
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if child.lineno > header_end:
                    parts.append(self._slice(_start(child), child.end_lineno))

        outline = []
        for child in members:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                outline.append(
                    f"    {self._signature(child)}  # L{_start(child)}-{child.end_lineno}"
                )
            elif isinstance(child, ast.ClassDef):
                outline.append(f"    class {child.name}  # L{_start(child)}-{child.end_lineno}")
        text = "\n".join(parts)
        if outline:
            text += "\n\n# methods:\n" + "\n".join(outline)

        bases = [ast.unparse(b) for b in node.bases]
        bases += [
            f"{k.arg}={ast.unparse(k.value)}" if k.arg else f"**{ast.unparse(k.value)}"
            for k in node.keywords
        ]
        signature = f"class {node.name}" + (f"({', '.join(bases)})" if bases else "")

        class_unit = CodeUnit(
            kind="class", name=qname, start_line=start, end_line=end, text=text,
            signature=signature, parent=parent,
            decorators=tuple(ast.unparse(d) for d in node.decorator_list),
            bases=tuple(bases),
            docstring=ast.get_docstring(node),
            imports_used=self._imports_used(node, shallow=True),
        )

        units = [class_unit]
        prev_end = header_end
        for child in members:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                units.append(self._function_unit(child, qname, prev_end, qname))
            elif isinstance(child, ast.ClassDef):
                units.extend(self._class_units(child, qname, prev_end))
            prev_end = child.end_lineno
        return units

    @staticmethod
    def _signature(node) -> str:
        prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
        signature = f"{prefix}def {node.name}({ast.unparse(node.args)})"
        if node.returns is not None:
            signature += f" -> {ast.unparse(node.returns)}"
        return signature

    def _function_unit(self, node, parent, floor: int, enclosing_class) -> CodeUnit:
        start = self._with_leading_comments(_start(node), floor)
        end = node.end_lineno
        name = f"{parent}.{node.name}" if parent else node.name
        return CodeUnit(
            kind="method" if parent else "function",
            name=name, start_line=start, end_line=end,
            text=self._slice(start, end),
            signature=self._signature(node), parent=parent,
            decorators=tuple(ast.unparse(d) for d in node.decorator_list),
            docstring=ast.get_docstring(node),
            imports_used=self._imports_used(node),
            calls=self._calls(node, enclosing_class, name),
            is_async=isinstance(node, ast.AsyncFunctionDef),
        )

    

    def _imports_used(self, node, shallow: bool = False) -> tuple:
        
        if shallow:
            parts = list(node.bases) + list(node.decorator_list) + [
                c for c in node.body
                if not isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            ]
            walkers = [n for part in parts for n in ast.walk(part)]
        else:
            walkers = list(ast.walk(node))
        names = {n.id for n in walkers if isinstance(n, ast.Name)}
        return tuple(sorted({self.imports[n] for n in names if n in self.imports}))

    def _calls(self, node, enclosing_class, own_name: str) -> tuple:
        found = []
        for n in ast.walk(node):
            if not isinstance(n, ast.Call):
                continue
            func = n.func
            if isinstance(func, ast.Name):
                if func.id in self.top_symbols:
                    found.append(func.id)
            elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                base = func.value.id
                if base in ("self", "cls") and enclosing_class:
                    if func.attr in self.class_methods.get(enclosing_class, ()):
                        found.append(f"{enclosing_class}.{func.attr}")
                elif base in self.class_methods and func.attr in self.class_methods[base]:
                    found.append(f"{base}.{func.attr}")
        return tuple(c for c in _dedupe(found) if c != own_name)
