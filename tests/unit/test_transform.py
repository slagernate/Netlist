
from netlist.transform import MapBSIM4ModelParams
from netlist.data import (
    Program,
    SourceFile,
    ModelDef,
    ParamDecl,
    Ident,
    Float,
    Ref,
    SubcktDef,
)

def test_map_bsim4_params():
    """Test that MapBSIM4ModelParams correctly renames deltox to dtox in BSIM4 models."""
    
    # Create a BSIM4 model with deltox
    bsim4_model = ModelDef(
        name=Ident("nmos_bsim4"),
        mtype=Ident("bsim4"),
        args=[],
        params=[
            ParamDecl(name=Ident("type"), default=Ref(ident=Ident("n")), distr=None),
            ParamDecl(name=Ident("deltox"), default=Float(1e-9), distr=None),
            ParamDecl(name=Ident("version"), default=Float(4.5), distr=None)
        ]
    )
    
    # Create a non-BSIM4 model with deltox (should NOT be mapped)
    other_model = ModelDef(
        name=Ident("nmos_other"),
        mtype=Ident("other"),
        args=[],
        params=[
            ParamDecl(name=Ident("deltox"), default=Float(1e-9), distr=None),
        ]
    )
    
    # Create a subcircuit containing a BSIM4 model
    subckt = SubcktDef(
        name=Ident("mysub"),
        ports=[],
        params=[],
        entries=[bsim4_model] # Nested model
    )

    program = Program(files=[
        SourceFile(path="test.scs", contents=[bsim4_model, other_model, subckt])
    ])

    # Run the pass
    pass_instance = MapBSIM4ModelParams()
    new_program = pass_instance.run(program)
    
    # Check top-level BSIM4 model
    new_bsim4 = new_program.files[0].contents[0]
    param_names = [p.name.name for p in new_bsim4.params]
    assert "deltox" not in param_names
    assert "dtox" in param_names
    
    # Check non-BSIM4 model
    new_other = new_program.files[0].contents[1]
    param_names_other = [p.name.name for p in new_other.params]
    assert "deltox" in param_names_other
    assert "dtox" not in param_names_other

    # Check nested BSIM4 model inside subcircuit
    new_subckt = new_program.files[0].contents[2]
    new_nested_bsim4 = new_subckt.entries[0]
    nested_param_names = [p.name.name for p in new_nested_bsim4.params]
    assert "deltox" not in nested_param_names
    assert "dtox" in nested_param_names

