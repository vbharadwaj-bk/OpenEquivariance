from typing import NamedTuple, Literal, get_args

from src.implementations.e3nn_lite import TPProblem, Instruction
from src.benchmark.e3nn_lite_utils import calc_weight_offsets, CGTensor, RepData


Dimension = Literal['in1', 'in2', 'out']

class InstructionInfo(NamedTuple):
    """
    This is a class that provides a superset of the information in an Instruction to make templating easier
    """
    # Index Values (Not actually useful)
    in1_index : int
    in2_index : int
    out_index : int
    # Offsets from the base pointer
    in1_offset : int 
    in2_offset : int 
    out_offset : int 
    weight_offset : int
    # Orders 
    in1_l : int 
    in2_l : int 
    out_l : int
    # Multiplicities
    in1_multiplicity : int
    in2_multiplicity : int
    out_multiplicity : int
    # Irrep Length 
    in1_irrep_length : int
    in2_irrep_length : int 
    out_irrep_length : int
    # Tensor Info
    tensor : CGTensor
    path_weight: float
    # Legacy Info (should be accounted for before the templating level)
    connection_mode: str
    has_weight: bool
    path_shape: tuple
    # Weight Sub Partitioning Info
    weight_in1_extent : int
    weight_in2_extent : int 
    weight_out_extent : int 
    weight_in1_offset : int
    weight_in2_offset : int 
    weight_out_offset : int 

def prepare_InstructionInfo_list(problem : TPProblem) -> list[InstructionInfo]:
    """
    This is a convenience funtion that wraps all the info needed at the C++ level into one object
    """
    infolist = []
    L1 = RepData(problem.irreps_in1) 
    L2 = RepData(problem.irreps_in2) 
    L3 = RepData(problem.irreps_out) 
    weight_offsets = calc_weight_offsets(problem)
    assert isinstance(weight_offsets, list)
    assert len(weight_offsets) == len(list(problem.instructions))
    ins : Instruction
    for ins_index, ins in enumerate(problem.instructions): 
        infolist.append(
            InstructionInfo(
                # Irrep Indices 
                in1_index=ins.i_in1,
                in2_index=ins.i_in2,
                out_index=ins.i_out, 
                # Offsets
                in1_offset=L1.offsets[ins.i_in1],
                in2_offset=L2.offsets[ins.i_in2],
                out_offset=L3.offsets[ins.i_out], 
                weight_offset=weight_offsets[ins_index],
                # Orders
                in1_l=L1.ls[ins.i_in1],
                in2_l=L2.ls[ins.i_in2],
                out_l=L3.ls[ins.i_out],
                # Multiplicites
                in1_multiplicity=L1.mults[ins.i_in1],
                in2_multiplicity=L2.mults[ins.i_in2],
                out_multiplicity=L3.mults[ins.i_out],
                # Irrep Length 
                in1_irrep_length=L1.irrep_lengths[ins.i_in1],
                in2_irrep_length=L2.irrep_lengths[ins.i_in2],
                out_irrep_length=L3.irrep_lengths[ins.i_out],
                # Tensor Info 
                tensor=CGTensor(L1.ls[ins.i_in1], L2.ls[ins.i_in2], L3.ls[ins.i_out]),
                path_weight=ins.path_weight,
                # Legacy Info 
                connection_mode=ins.connection_mode,
                has_weight=ins.has_weight,
                path_shape=ins.path_shape,
                # Weight Sub Partitioning Info
                weight_in1_extent=L1.mults[ins.i_in1],
                weight_in2_extent=L2.mults[ins.i_in2],
                weight_out_extent=L3.mults[ins.i_out],
                weight_in1_offset=0, 
                weight_in2_offset=0, 
                weight_out_offset=0, 
            )
        )
    return infolist

def partition_InstructionInfo_list_by_max_size_along_dimension(input_II_list : list[InstructionInfo], max_size : int, dimension : Dimension) -> list[InstructionInfo]: 
    assert dimension in get_args(Dimension)
    output_II_list = []
    while input_II_list:
        II = input_II_list.pop()
        multiplicity = getattr(II,f"{dimension}_multiplicity")
        assert isinstance(multiplicity, int)

        if multiplicity > max_size:
            # hunk is the max_sized bit 
            # rest is the rest of it   

            irrep_offsets : dict[Dimension, int]= {
                'in1' : II.in1_offset, 
                'in2' : II.in2_offset, 
                'out' : II.out_offset, 
            }
            hunk_irrep_offsets = irrep_offsets.copy()
            rest_irrep_offsets = irrep_offsets.copy()

            irrep_length = getattr(II,f"{dimension}_irrep_length")
            assert isinstance(irrep_length, int)
            rest_irrep_offsets[dimension] += max_size * irrep_length

            weight_offsets : dict[Dimension, int]= {
                'in1' : II.weight_in1_offset,
                'in2' : II.weight_in2_offset,
                'out' : II.weight_out_offset, 
            }
            hunk_weight_offsets = weight_offsets.copy()
            rest_weight_offsets = weight_offsets.copy() 

            rest_weight_offsets[dimension] += max_size

            multiplicities : dict[Dimension, int] = {
                'in1' : II.in1_multiplicity,
                'in2' : II.in2_multiplicity, 
                'out' : II.out_multiplicity, 
            }
            hunk_multiplicities = multiplicities.copy()
            rest_multiplicities = multiplicities.copy()

            hunk_multiplicities[dimension]  = max_size
            rest_multiplicities[dimension] -= max_size 

            rest_II = InstructionInfo(
                # Irrep Indices 
                in1_index=II.in1_index, # This won't acutally be accurate with the partition, but it will correspond to the original blocks
                in2_index=II.in2_index,
                out_index=II.out_index, 
                # Offsets
                in1_offset=rest_irrep_offsets['in1'],
                in2_offset=rest_irrep_offsets['in2'],
                out_offset=rest_irrep_offsets['out'], 
                weight_offset=II.weight_offset,
                # Orders
                in1_l=II.in1_l,
                in2_l=II.in2_l,
                out_l=II.out_l,
                # Multiplicites
                in1_multiplicity=rest_multiplicities['in1'],
                in2_multiplicity=rest_multiplicities['in2'],
                out_multiplicity=rest_multiplicities['out'],
                # Irrep Length 
                in1_irrep_length=II.in1_irrep_length,
                in2_irrep_length=II.in2_irrep_length,
                out_irrep_length=II.out_irrep_length,
                # Tensor Info 
                tensor=II.tensor,
                path_weight=II.path_weight,
                # Legacy Info 
                connection_mode=II.connection_mode,
                has_weight=II.has_weight,
                path_shape=(rest_multiplicities['in1'], rest_multiplicities['in2'], rest_multiplicities['out']),
                # Weight Sub Partitioning Info
                weight_in1_extent=II.weight_in1_extent,
                weight_in2_extent=II.weight_in2_extent,
                weight_out_extent=II.weight_out_extent,
                weight_in1_offset=rest_weight_offsets['in1'], 
                weight_in2_offset=rest_weight_offsets['in2'], 
                weight_out_offset=rest_weight_offsets['out'], 
            )

            hunk_II = InstructionInfo(
                # Irrep Indices 
                in1_index=II.in1_index, # This won't acutally be accurate with the partition, but it will correspond to the original blocks
                in2_index=II.in2_index,
                out_index=II.out_index, 
                # Offsets
                in1_offset=hunk_irrep_offsets['in1'],
                in2_offset=hunk_irrep_offsets['in2'],
                out_offset=hunk_irrep_offsets['out'], 
                weight_offset=II.weight_offset,
                # Orders
                in1_l=II.in1_l,
                in2_l=II.in2_l,
                out_l=II.out_l,
                # Multiplicites
                in1_multiplicity=hunk_multiplicities['in1'],
                in2_multiplicity=hunk_multiplicities['in2'],
                out_multiplicity=hunk_multiplicities['out'],
                # Irrep Length 
                in1_irrep_length=II.in1_irrep_length,
                in2_irrep_length=II.in2_irrep_length,
                out_irrep_length=II.out_irrep_length,
                # Tensor Info 
                tensor=II.tensor,
                path_weight=II.path_weight,
                # Legacy Info 
                connection_mode=II.connection_mode,
                has_weight=II.has_weight,
                path_shape=(hunk_multiplicities['in1'], hunk_multiplicities['in2'], hunk_multiplicities['out']),
                # Weight Sub Partitioning Info
                weight_in1_extent=II.weight_in1_extent,
                weight_in2_extent=II.weight_in2_extent,
                weight_out_extent=II.weight_out_extent,
                weight_in1_offset=hunk_weight_offsets['in1'], 
                weight_in2_offset=hunk_weight_offsets['in2'], 
                weight_out_offset=hunk_weight_offsets['out'], 
            )
            output_II_list.append(hunk_II)
            input_II_list.append(rest_II)   
        else: 
            output_II_list.append(II)      
    return output_II_list

def pretty_format_InstructionInfoList(II_list : list[InstructionInfo]) -> str:
    s = "\n"
    s += "|      index      |       l         |        m        |  irrep offset   |      weight offset     |  weight extent  |\n"
    s += "| in1 | in2 | out | in1 | in2 | out | in1 | in2 | out | in1 | in2 | out | base | in1 | in2 | out | in1 | in2 | out |\n"
    for II in II_list: 
        s += f"| {II.in1_index:3} | {II.in2_index:3} | {II.out_index:3} |"
        s += f" {II.in1_l:3} | {II.in2_l:3} | {II.out_l:3} |"
        s += f" {II.in1_multiplicity:3} | {II.in2_multiplicity:3} | {II.out_multiplicity:3} |"
        s += f" {II.in1_offset:3} | {II.in2_offset:3} | {II.out_offset:3} |"
        s += f" {II.weight_offset:4} |"
        s += f" {II.weight_in1_offset:3} | {II.weight_in2_offset:3} | {II.weight_out_offset:3} |"
        s += f" {II.weight_in1_extent:3} | {II.weight_in2_extent:3} | {II.weight_out_extent:3} |"
        s += "\n"
    return s 