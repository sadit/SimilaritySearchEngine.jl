@testset "a filtered search spends the candidate budget it was given, and says so by returning fewer" begin
    mktempworkdir() do workdir
        # Exact backend and vectors ranked by construction: item i sits at distance i-1 from
        # the query, so "the item the predicate wants" has a known rank.
        h = create_project(workdir, "budget_ds", DenseVectors(; index=Exhaustive()))
        append_items!(h, [DenseItem(Float32[i - 1, 0, 0, 0]; doc_id="v$i") for i in 1:64])
        q = Float32[0, 0, 0, 0]

        wants_the_farthest = (record, meta) -> record.doc_id == "v64"
        # Default budget is 2k = 2 candidates: the item ranked 64th is not among them.
        @test isempty(search(h, q, 1; filter=wants_the_farthest))
        # Paying for the whole collection finds it. This is the documented trade-off.
        @test [r.doc_id for r in search(h, q, 1; filter=wants_the_farthest, candidates=64)] == ["v64"]

        # A predicate most items satisfy is served by the default budget.
        @test length(search(h, q, 5; filter=(record, meta) -> true)) == 5
        close_project!(h)
    end
end

@testset "a declared meta schema says what a field holds" begin
    # The declaration itself: what it accepts, and what it refuses to be
    @test isempty(MetaSchema())
    @test_throws InvalidOption MetaField("year", :int32)
    @test_throws InvalidOption MetaField("", :int64)

    sch = MetaSchema([MetaField("year", :int64), MetaField("lang", :string),
                      MetaField("score", :float64), MetaField("ok", :bool),
                      MetaField("when", :timestamp)])

    # A value of the declared type is stored as that type, and a conversion that keeps the
    # value is applied: JSON has one number, and 2024.0 is the integer 2024.
    m = Schema.coerce_meta(sch, Dict("year" => 2024.0, "score" => 3, "ok" => true,
                                     "lang" => "es", "when" => "2026-09-13T10:00:00",
                                     "undeclared" => [1, 2]))
    @test m["year"] === Int64(2024)
    @test m["score"] === 3.0
    @test m["lang"] == "es"
    @test m["when"] == "2026-09-13T10:00:00"
    # Declaring is additive: a field nobody declared is stored as it arrived
    @test m["undeclared"] == [1, 2]

    # A value that is not the declared type is an invalid request, not a stored surprise
    @test_throws PayloadMismatch Schema.coerce_meta(sch, Dict("year" => "2024"))
    @test_throws PayloadMismatch Schema.coerce_meta(sch, Dict("year" => 2024.5))
    @test_throws PayloadMismatch Schema.coerce_meta(sch, Dict("ok" => 1))
    @test_throws PayloadMismatch Schema.coerce_meta(sch, Dict("when" => "last tuesday"))
    @test_throws PayloadMismatch Schema.coerce_meta(sch, Dict("lang" => 3))

    # A record need not carry every declared field, and a null is an absent field
    @test !haskey(Schema.coerce_meta(sch, Dict("lang" => "es")), "year")
    @test !haskey(Schema.coerce_meta(sch, Dict("year" => nothing)), "year")

    # A project that declares nothing pays nothing: the same object comes back
    untouched = Dict("year" => "2024")
    @test Schema.coerce_meta(MetaSchema(), untouched) === untouched
    @test Schema.coerce_meta(nothing, untouched) === untouched

    @test Schema.declared_type(sch, "year") === :int64
    @test Schema.declared_type(sch, "nope") === nothing
end

@testset "a declared type is what a filter compares in" begin
    record = Schema.MetadataRecord(Int32(1), 1, "d1", String[], String[])
    sch = MetaSchema([MetaField("when", :timestamp)])
    meta = Dict("when" => "2026-01-02T00:00:00")

    # The same instant written two ways: equal as instants, different as text
    @test Schema.matches_filter(record, meta, Dict("when" => "2026-01-02T00:00"), sch)
    @test !Schema.matches_filter(record, meta, Dict("when" => "2026-01-02T00:00"), nothing)

    @test Schema.matches_filter(record, meta, Dict("when" => Dict("lt" => "2026-02-01T00:00:00")), sch)
    @test !Schema.matches_filter(record, meta, Dict("when" => Dict("gt" => "2026-02-01T00:00:00")), sch)
    # An undeclared field compares exactly as it did before
    @test Schema.matches_filter(record, Dict("n" => 5), Dict("n" => Dict("gte" => 3)), sch)
end

@testset "a project declares its meta schema, and declaring later leaves old records alone" begin
    mktempworkdir() do workdir
        h = create_project(workdir, "schema_ds", DenseVectors(; index=Exhaustive()); meta_schema=["year" => :int64, "lang" => :string])
        @test [f.name for f in meta_schema(h).fields] == ["year", "lang"]
        @test meta_schema(h).version == 1

        append_items!(h, [DenseItem(Float32[1, 0, 0, 0]; doc_id="a",
                                    meta=Dict{String,Any}("year" => 2024.0, "lang" => "es", "free" => "kept"))])
        it = only(fetch_items(h, ["a"]))
        @test it.meta["year"] === Int64(2024)      # converted on the way in
        @test it.meta["free"] == "kept"            # undeclared, stored as it arrived

        @test_throws PayloadMismatch append_items!(h, [DenseItem(Float32[2, 0, 0, 0]; doc_id="b",
                                                                 meta=Dict{String,Any}("year" => "2024"))])

        # A value written before a field was declared is not read and not checked
        append_items!(h, [DenseItem(Float32[3, 0, 0, 0]; doc_id="c",
                                    meta=Dict{String,Any}("score" => "high"))])
        declare_meta_schema!(h, ["year" => :int64, "lang" => :string, "score" => :float64])
        @test meta_schema(h).version == 2
        @test only(fetch_items(h, ["c"])).meta["score"] == "high"
        # ... while what is written from now on is checked
        @test_throws PayloadMismatch append_items!(h, [DenseItem(Float32[4, 0, 0, 0]; doc_id="d",
                                                                 meta=Dict{String,Any}("score" => "high"))])
        close_project!(h)

        # The declaration is part of the project, not of the handle that made it
        h2 = open_project(workdir, "schema_ds")
        @test meta_schema(h2).version == 2
        @test [f.name for f in meta_schema(h2).fields] == ["year", "lang", "score"]
        @test only(fetch_items(h2, ["a"])).meta["year"] === Int64(2024)
        close_project!(h2)
    end
end
