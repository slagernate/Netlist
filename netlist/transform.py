from abc import ABC, abstractmethod
from typing import List, Optional, Set
from dataclasses import replace

from netlist.data import (
    Program,
    SourceFile,
    ModelDef,
    ModelFamily,
    ModelVariant,
    ParamDecl,
    Ident,
    Instance,
    SubcktDef,
    Entry,
)

class Pass(ABC):
    """Base class for AST transformation passes."""
    
    @abstractmethod
    def run(self, program: Program) -> Program:
        """Run the pass on the given program, returning a new (or modified) program."""
        pass


class MapBSIM4ModelParams(Pass):
    """
    Maps parameters in BSIM4 model definitions to their target dialect equivalents.
    For example, maps 'deltox' to 'dtox' for Xyce compatibility.
    """
    
    def __init__(self, target_dialect: str = "xyce"):
        self.target_dialect = target_dialect

    def run(self, program: Program) -> Program:
        new_files = []
        for file in program.files:
            new_contents = self._transform_entries(file.contents)
            new_files.append(replace(file, contents=new_contents))
        return replace(program, files=new_files)

    def _transform_entries(self, entries: List[Entry]) -> List[Entry]:
        new_entries = []
        for entry in entries:
            if isinstance(entry, ModelDef):
                new_entries.append(self._transform_model_def(entry))
            elif isinstance(entry, ModelFamily):
                new_entries.append(self._transform_model_family(entry))
            elif isinstance(entry, SubcktDef):
                # Recurse into subcircuits
                new_entries.append(self._transform_subckt(entry))
            else:
                new_entries.append(entry)
        return new_entries

    def _transform_subckt(self, subckt: SubcktDef) -> SubcktDef:
        new_entries = self._transform_entries(subckt.entries)
        return replace(subckt, entries=new_entries)

    def _transform_model_def(self, model: ModelDef) -> ModelDef:
        if model.mtype.name.lower() == "bsim4":
            return self._map_params(model)
        return model

    def _transform_model_family(self, family: ModelFamily) -> ModelFamily:
        if family.mtype.name.lower() == "bsim4":
            new_variants = []
            for variant in family.variants:
                # ModelVariant has params too
                new_params = self._process_param_list(variant.params)
                new_variants.append(replace(variant, params=new_params))
            return replace(family, variants=new_variants)
        return family

    def _map_params(self, model: ModelDef) -> ModelDef:
        new_params = self._process_param_list(model.params)
        return replace(model, params=new_params)

    def _process_param_list(self, params: List[ParamDecl]) -> List[ParamDecl]:
        new_params = []
        for p in params:
            if p.name.name == "deltox":
                # Map deltox -> dtox
                new_params.append(replace(p, name=Ident("dtox")))
            else:
                new_params.append(p)
        return new_params

