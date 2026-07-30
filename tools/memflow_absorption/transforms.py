"""Frozen transform registry and executable route fixtures."""
from __future__ import annotations
import hashlib, importlib, json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from memo.operational_event import canonical_json_bytes

class TransformRegistryError(ValueError): pass

@dataclass(frozen=True)
class TransformSpec:
    transform_id: str
    module: str
    implementation_sha256: str
    input_schema_id: str
    output_schema_id: str
    version: str

@dataclass(frozen=True)
class RouteFixture:
    fixture_id: str
    request: dict[str, Any]
    expected_result: Any = None
    expected_error: Any = None
    def to_dict(self):
        d={'fixture_id':self.fixture_id,'request':self.request}
        if self.expected_error is not None: d['expected_error']=self.expected_error
        else: d['expected_result']=self.expected_result
        return d

def implementation_digest(module: str) -> str:
    mod=importlib.import_module(module)
    path=Path(mod.__file__ or '').resolve()
    return hashlib.sha256(path.read_bytes()).hexdigest()

class FrozenTransformRegistry:
    def __init__(self, specs=()):
        self._specs={s.transform_id:s for s in specs}
    def register(self,spec:TransformSpec):
        if spec.transform_id in self._specs: raise TransformRegistryError('duplicate transform_id')
        if implementation_digest(spec.module)!=spec.implementation_sha256: raise TransformRegistryError('implementation digest mismatch')
        self._specs[spec.transform_id]=spec
    def resolve(self, transform_id: str) -> Callable[...,Any]:
        try: s=self._specs[transform_id]
        except KeyError as e: raise TransformRegistryError(f'unknown transform: {transform_id}') from e
        if implementation_digest(s.module)!=s.implementation_sha256: raise TransformRegistryError('implementation digest mismatch')
        fn=getattr(importlib.import_module(s.module),'transform',None)
        if not callable(fn): raise TransformRegistryError('transform implementation missing')
        return fn
    def to_dict(self): return [s.__dict__ for s in sorted(self._specs.values(), key=lambda x:x.transform_id)]
    def digest(self): return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()

def load_fixture(path: Path) -> RouteFixture:
    d=json.loads(path.read_bytes())
    if not isinstance(d,dict) or 'request' not in d or (('expected_result' in d)+('expected_error' in d))!=1:
        raise TransformRegistryError('fixture requires request and exactly one expected outcome')
    return RouteFixture(d.get('fixture_id',path.name),d['request'],d.get('expected_result'),d.get('expected_error'))

def evaluate_predicate(predicate: dict[str,Any], request: dict[str,Any]) -> bool:
    for key, matcher in predicate.items():
        value=request.get(key)
        if 'equals' in matcher and value!=matcher['equals']: return False
        if 'in' in matcher and value not in matcher['in']: return False
        if 'exists' in matcher and (key in request)!=matcher['exists']: return False
    return True

def verify_route_fixtures(routes, registry: FrozenTransformRegistry, root: Path):
    seen=set()
    for route in routes:
        for rel in route.fixture_paths:
            if rel in seen: raise TransformRegistryError('fixture covered by multiple routes')
            seen.add(rel); fixture=load_fixture(root/rel)
            if not evaluate_predicate(dict(route.predicate),fixture.request): raise TransformRegistryError('fixture predicate mismatch')
            fn=registry.resolve(route.transform_id)
            try: actual=fn(fixture.request)
            except Exception as exc:
                if fixture.expected_error is None or type(exc).__name__!=fixture.expected_error: raise TransformRegistryError('fixture expected error mismatch')
            else:
                if fixture.expected_error is not None or actual!=fixture.expected_result: raise TransformRegistryError('fixture result mismatch')
