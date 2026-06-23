from brainhops.datamodel.transformations import Affine, DisplacementField, CoordinatesField, Sequence, Permutation, Scaling, Translation, Linear
import numpy as np
import dask.array as da


def test_affine_with_coordinates():
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
        [[2, 1, 0, 100], [1, 2, 1, 5], [0, 0, 1, 30]]))
    sequence = Sequence(transformations=[affine, field])
    assert isinstance(sequence.transformations[1].field, da.Array)
    output = sequence.compute()
    assert isinstance(output.field, da.Array)
    assert not isinstance(output, da.Array)
    assert output.field[1, 1, 1, 0] == 103
    assert output.field[1, 1, 1, 1] == 9
    assert output.field[1, 1, 1, 2] == 31
    assert output.field[1, 4, 1, 0] == 106
    assert output.field[1, 4, 1, 1] == 15
    assert output.field[1, 4, 1, 2] == 31


def test_linear_with_coordinates():
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
    sequence = Sequence(transformations=[linear, field])
    output = sequence.compute()
    assert output.field[1, 1, 1, 0] == 3
    assert output.field[1, 1, 1, 1] == 4
    assert output.field[1, 1, 1, 2] == 1
    assert output.field[1, 4, 1, 0] == 6
    assert output.field[1, 4, 1, 1] == 10
    assert output.field[1, 4, 1, 2] == 1


def test_displacement_with_coordinates():
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
    sequence = Sequence(transformations=[displacement, field])
    output = sequence.compute()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                for i in range(3):
                    assert output.field[x, y, z, i] == identity[x, y, z, i]*2


def test_affine_displacement_coordinates():
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
        [[2, 1, 0, 0], [1, 2, 1, 0], [0, 0, 1, 0]]))
    sequence = Sequence(transformations=[affine, displacement, field])
    input_tensor = np.zeros((100, 100, 100, 1))
    input_tensor[18, 24, 6, 0] = 1
    input_tensor[20, 18, 2, 0] = 2
    output_tester = np.zeros((10, 10, 10))
    output_tester[3, 3, 3] = 1
    output_tester[4, 2, 1] = 2
    input_coord = CoordinatesField(field=input_tensor)
    output = sequence(input_coord, True)
    for x in range(10):
        for y in range(10):
            for z in range(10):
                assert output.field[x, y, z, 0] == output_tester[x, y, z]


def test_permutation_with_field():
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
    sequence = Sequence(transformations=[permutation, field])
    output = sequence.compute()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                for i in range(3):
                    assert output.field[x, y, z, i] == identity[y, z, x, i]


def test_scale_with_coordinates():
    g_vals = np.meshgrid(
        *[np.arange(s) for s in [5, 5, 5]],
        indexing="ij"
    )
    identity = np.ones(
        (5, 5, 5, 3))
    for i in range(len(g_vals)):
        identity[:, :, :, i] = g_vals[i]
    field = CoordinatesField(field=identity)
    scale = Scaling(scale=[1, 2, 0])
    sequence = Sequence(transformations=[scale, field])
    output = sequence.compute()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                assert output.field[x, y, z, 0] == identity[x, y, z, 0]*1
                assert output.field[x, y, z, 1] == identity[x, y, z, 1]*2
                assert output.field[x, y, z, 2] == identity[x, y, z, 2]*0


def test_translation_with_coordinates():
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
    sequence = Sequence(transformations=[scale, field])
    output = sequence.compute()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                assert output.field[x, y, z, 0] == identity[x, y, z, 0]+1
                assert output.field[x, y, z, 1] == identity[x, y, z, 1]+2
                assert output.field[x, y, z, 2] == identity[x, y, z, 2]+0


def test_combinations_of_linear_transformations():
    linear_mat = np.array([[1, 2, 5], [3, 8, 4], [1, 1, 1]])

    linear = Linear(matrix=linear_mat)
    scale = Scaling(scale=np.array([1, 2, 3]))
    sl = (scale @ linear).compute()
    sl_output = np.array([[1, 2, 5], [6, 16, 8], [3, 3, 3]])
    for i in range(3):
        for j in range(3):
            assert sl.matrix[i, j] == sl_output[i, j]

    translation = Translation(translation=np.array([4, 5, 6]))
    ts = (translation @ scale).compute()
    ts_output = np.array(
        [[1, 0, 0, 4], [0, 2, 0, 5], [0, 0, 3, 6]])

    for i in range(3):
        for j in range(4):
            assert ts.matrix[i, j] == ts_output[i, j]

    permutation = Permutation(permutation=np.array([2, 1, 0]))
    ps = (permutation @ scale).compute()

    ps_output = np.array([[0, 0, 3], [0, 2, 0], [1, 0, 0]])

    for i in range(3):
        for j in range(3):
            assert ps.matrix[i, j] == ps_output[i, j]
