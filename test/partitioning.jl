using Jutul, Test, SparseArrays, LinearAlgebra

@test Jutul.compress_partition([1, 3, 6, 5]) == [1, 2, 4, 3]
@test Jutul.compress_partition(collect(1:10)) == 1:10

function test_basic_partition_features(p, np)
    for i in 1:np
        @test count(isequal(i), p) > 0
    end
    @test minimum(p) == 1
    @test maximum(p) == np
end

m = LinearPartitioner()
A = sprand(100, 100, 0.2) + I
A = A + A'

for np = 1:10
    p = Jutul.partition(LinearPartitioner(), A, np)
    test_basic_partition_features(p, np)
    p = Jutul.partition(MetisPartitioner(), A, np)
    test_basic_partition_features(p, np)
end

l = 1:49
r = 2:50
N = vcat(l', r')

np = 5
p = Jutul.partition(N, np)
test_basic_partition_features(p, np)

function test_partition_groups(p, grps)
    for grp in grps
        pg = p[grp]
        @test all(isequal(first(pg)), pg)
    end
end

grps = [[3, 4, 5], [17, 19, 18]]
p = Jutul.partition(N, np, groups = grps, group_by_weights = false)
test_partition_groups(p, grps)

p = Jutul.partition(N, np, groups = grps, group_by_weights = true, buffer_group = true)
test_partition_groups(p, grps)

p = Jutul.partition(N, np, groups = grps, group_by_weights = true, buffer_group = false)
test_partition_groups(p, grps)
