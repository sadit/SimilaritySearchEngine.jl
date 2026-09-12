using JLD2
using SimilaritySearchEngine.IndexEngine

function save_snapshot(path::String, engine::IndexEngine.AbstractSearchEngine)
    jldsave(path; engine=engine)
end

function load_snapshot(path::String)
    f = jldopen(path, "r")
    try
        return f["engine"]
    finally
        close(f)
    end
end
