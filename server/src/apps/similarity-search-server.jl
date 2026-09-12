using SimilaritySearchServer

if abspath(PROGRAM_FILE) == @__FILE__
    exit(SimilaritySearchServer.main_serve(ARGS))
end
