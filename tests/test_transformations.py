from brainhops.datamodel.transforms import Affine, DisplacementField, CoordinatesField, Sequence, Permutation, Scale, Translation, Linear, CartesianSliceField
import numpy as np
import dask.array as da


def test_affine():
    g_vals = da.meshgrid(
        *[da.arange(s) for s in [5, 5, 5]],
        indexing="ij"
    )
    identity = da.ones(
        (5, 5, 5, 3))
    for i in range(len(g_vals)):
        identity[:, :, :, i] = g_vals[i]
    field = CoordinatesField(field=identity)
    affine = Affine(matrix=np.asarray(
        [[2, 1, 0, 100], [1, 2, 1, 5], [0, 0, 1, 30], [0, 0, 0, 1]]))
    sequence = Sequence(transforms=[affine, field])
    assert isinstance(sequence.transforms[1].field, da.Array)
    output = sequence.compute()
    assert isinstance(output.field, da.Array)
    assert not isinstance(output, da.Array)
    assert output.field[1, 1, 1, 0] == 103
    assert output.field[1, 1, 1, 1] == 9
    assert output.field[1, 1, 1, 2] == 31
    assert output.field[1, 4, 1, 0] == 106
    assert output.field[1, 4, 1, 1] == 15
    assert output.field[1, 4, 1, 2] == 31


def test_linear():
    g_vals = da.meshgrid(
        *[da.arange(s) for s in [5, 5, 5]],
        indexing="ij"
    )
    identity = da.ones(
        (5, 5, 5, 3))
    for i in range(len(g_vals)):
        identity[:, :, :, i] = g_vals[i]
    field = CoordinatesField(field=identity)
    linear = Linear(matrix=np.asarray(
        [[2, 1, 0], [1, 2, 1], [0, 0, 1]]))
    sequence = Sequence(transforms=[linear, field])
    output = sequence.compute()
    assert output.field[1, 1, 1, 0] == 3
    assert output.field[1, 1, 1, 1] == 4
    assert output.field[1, 1, 1, 2] == 1
    assert output.field[1, 4, 1, 0] == 6
    assert output.field[1, 4, 1, 1] == 10
    assert output.field[1, 4, 1, 2] == 1


def test_displacement():
    g_vals = np.meshgrid(
        *[np.arange(s) for s in [5, 5, 5]],
        indexing="ij"
    )
    identity = np.ones(
        (5, 5, 5, 3))
    for i in range(len(g_vals)):
        identity[:, :, :, i] = g_vals[i]
    field = CoordinatesField(field=identity)
    displacement = DisplacementField(field=identity)
    sequence = Sequence(transforms=[displacement, field])
    output = sequence.compute()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                for i in range(3):
                    assert output.field[x, y, z, i] == identity[x, y, z, i]*2


def test_full_output():
    g_vals = np.meshgrid(
        *[np.arange(s) for s in [10, 10, 10]],
        indexing="ij"
    )
    identity = np.ones(
        (10, 10, 10, 3))
    for i in range(len(g_vals)):
        identity[:, :, :, i] = g_vals[i]
    field = CoordinatesField(field=identity)
    displacement = DisplacementField(field=identity)
    affine = Affine(matrix=np.asarray(
        [[2, 1, 0, 0], [1, 2, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
    sequence = Sequence(transforms=[affine, displacement, field])
    input_tensor = np.zeros((100, 100, 100))
    input_tensor[18, 24, 6] = 1
    input_tensor[20, 18, 2] = 2
    output_tester = np.zeros((10, 10, 10))
    output_tester[3, 3, 3] = 1
    output_tester[4, 2, 1] = 2
    input_coord = CoordinatesField(field=input_tensor)
    output = sequence(input_coord, True)
    for x in range(10):
        for y in range(10):
            for z in range(10):
                assert output.field[x, y, z] == output_tester[x, y, z]

    output = (CartesianSliceField(
        slices=(slice(2, 5), slice(2, 5), slice(2, 5)))
        @ output).compute()

    for x in range(3):
        for y in range(3):
            for z in range(3):
                assert output.field[x, y, z] == output_tester[x+2, y+2, z+2]


def test_full_output_soft():
    g_vals = np.meshgrid(
        *[np.arange(s) for s in [10, 10, 10]],
        indexing="ij"
    )
    identity = np.ones(
        (10, 10, 10, 3))
    for i in range(len(g_vals)):
        identity[:, :, :, i] = g_vals[i]
    field = CoordinatesField(field=identity)
    displacement = DisplacementField(field=identity)
    affine = Affine(matrix=np.asarray(
        [[2, 1, 0, 0], [1, 2, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
    linear = Linear(matrix=np.asarray([[2, 1, 0], [1, 2, 1], [0, 0, 1]]))
    sequence = Sequence(
        transforms=[linear, affine, affine, displacement, field])
    input_tensor = np.zeros((100, 100, 100))
    input_coord = CoordinatesField(field=input_tensor)
    output = sequence(input_coord, False)
    assert isinstance(output, Sequence)
    assert isinstance(output.transforms[0], CoordinatesField)
    assert isinstance(output.transforms[1], Affine)
    assert isinstance(output.transforms[2], DisplacementField)
    assert isinstance(output.transforms[3], CoordinatesField)
    assert len(output.transforms) == 4


def test_permutation():
    g_vals = np.meshgrid(
        *[np.arange(s) for s in [5, 5, 5]],
        indexing="ij"
    )
    identity = np.ones(
        (5, 5, 5, 3))
    for i in range(len(g_vals)):
        identity[:, :, :, i] = g_vals[i]
    field = CoordinatesField(field=identity)
    permutation = Permutation(permutation=[1, 2, 0])
    sequence = Sequence(transforms=[permutation, field])
    output = sequence.compute()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                for i in range(3):
                    assert output.field[y, z, x, i] == identity[x, y, z, i]


def test_scale():
    g_vals = np.meshgrid(
        *[np.arange(s) for s in [5, 5, 5]],
        indexing="ij"
    )
    identity = np.ones(
        (5, 5, 5, 3))
    for i in range(len(g_vals)):
        identity[:, :, :, i] = g_vals[i]
    field = CoordinatesField(field=identity)
    scale = Scale(scale=[1, 2, 0])
    sequence = Sequence(transforms=[scale, field])
    output = sequence.compute()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                assert output.field[x, y, z, 0] == identity[x, y, z, 0]*1
                assert output.field[x, y, z, 1] == identity[x, y, z, 1]*2
                assert output.field[x, y, z, 2] == identity[x, y, z, 2]*0


def test_translation():
    g_vals = np.meshgrid(
        *[np.arange(s) for s in [5, 5, 5]],
        indexing="ij"
    )
    identity = np.ones(
        (5, 5, 5, 3))
    for i in range(len(g_vals)):
        identity[:, :, :, i] = g_vals[i]
    field = CoordinatesField(field=identity)
    scale = Translation(translation=[1, 2, 0])
    sequence = Sequence(transforms=[scale, field])
    output = sequence.compute()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                assert output.field[x, y, z, 0] == identity[x, y, z, 0]+1
                assert output.field[x, y, z, 1] == identity[x, y, z, 1]+2
                assert output.field[x, y, z, 2] == identity[x, y, z, 2]+0


def test_combinations():
    linear_mat = np.array([[1, 2, 5], [3, 8, 4], [1, 1, 1]])

    linear = Linear(matrix=linear_mat)
    scale = Scale(scale=np.array([1, 2, 3]))
    sl = scale @ linear
    sl_output = np.array([[1, 2, 5], [6, 16, 8], [3, 3, 3]])
    for i in range(3):
        for j in range(3):
            assert sl.matrix[i, j] == sl_output[i, j]

    translation = Translation(translation=np.array([4, 5, 6]))
    ts = translation @ scale
    ts_output = np.array(
        [[1, 0, 0, 4], [0, 2, 0, 5], [0, 0, 3, 6], [0, 0, 0, 1]])

    for i in range(4):
        for j in range(4):
            assert ts.matrix[i, j] == ts_output[i, j]

    permutation = Permutation(permutation=np.array([2, 1, 0]))
    ps = permutation @ scale

    ps_output = np.array([[0, 0, 3], [0, 2, 0], [1, 0, 0]])

    for i in range(3):
        for j in range(3):
            assert ps.matrix[i, j] == ps_output[i, j]
