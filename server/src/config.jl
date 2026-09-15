# Configuration handling

"""
    generate_default_config() -> String

Generates a default TOML configuration with explanatory comments.

# Returns
- `String`: The default configuration content as a string.
"""
function generate_default_config()
    return """
# SimilaritySearchServer Configuration File
# This file contains all possible instance startup configurations.

[paths]
# Base working directory (relative to the execution directory or absolute)
workdir = "./workdir"
# Directory to store datasets (using RocksDB)
datasets = "datasets"
# Directory to store indices
indices = "indices"
# Spooling directory for asynchronous tasks (queued, running, done, failed)
spool = "jobs_spool"

[resources]
# How this server's threads are divided between answering queries and executing jobs.
#
# A job (allknn, fft, neardup, hsp, searchbatch, closestpair, build, dump, load) runs as its
# own process, so the reserved share is spent as a number of processes and a number of threads
# each: at most 4 jobs at once, each with `floor(threads * batch_threads_pct / 100) / 4`
# threads, and never more than the share in total. The rest of the threads stay with the
# server, which answers queries on them. `serve` prints the resulting numbers when it starts.
#
# Percentage of threads dedicated to live queries (0-100)
query_threads_pct = 80
# Percentage of threads dedicated to batch or heavy jobs. This is the one that applies when
# both are present and they do not add up to 100.
batch_threads_pct = 20
# How many searches may run at once. 0 derives it from the split above: the threads not
# reserved for jobs. A request that arrives when every slot is taken waits for one; `/metrics`
# reports how many are running, how many are waiting, and the time spent waiting, which is
# what says whether this bound is ever reached.
max_concurrent_queries = 0

[server]
# Listen address
host = "127.0.0.1"
# Server port
port = 8080

[search]
# Global policy in case of long queries: "deny" or "warn_only"
long_query_policy = "deny"

[auth]
# Require a token on every /api/v1 endpoint. /healthz, /readyz and /metrics stay open, so a
# supervisor and a metrics collector keep working without credentials.
#
# The default is false, which is what a server upgraded from an earlier version was doing
# already. With enabled = true the server refuses to start while no token exists, because in
# that state it could not answer any request; create the first one with
#
#     similarity-search-ctl add-token --user admin --permissions "admin:*"
#
# A permission is `operation:dataset`, where operation is read, write or admin, and dataset is
# a dataset id or `*`. read covers the queries and the GET endpoints, write covers append,
# delete, calibrate and job submission, admin covers dataset creation and deletion, the token
# endpoints and the control of jobs. Each operation includes the ones before it.
enabled = false
"""
end

"""
    load_config(path::String) -> Dict

Loads the TOML configuration from the given path.
If the file does not exist, it generates a default one.

# Arguments
- `path::String`: Path to the TOML configuration file.

# Returns
- `Dict`: The parsed TOML configuration dictionary.
"""
function load_config(path::String)
    if !isfile(path)
        println("Configuration file not found at ", path, ". Generating default configuration.")
        # Ensure directory exists if needed
        dir = dirname(path)
        if dir != "" && !isdir(dir)
            mkpath(dir)
        end
        open(path, "w") do io
            write(io, generate_default_config())
        end
    end
    
    return TOML.parsefile(path)
end
