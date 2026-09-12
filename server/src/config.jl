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
# Percentage of threads dedicated to live queries (0-100)
query_threads_pct = 80
# Percentage of threads dedicated to batch or heavy jobs
batch_threads_pct = 20

[server]
# Listen address
host = "127.0.0.1"
# Server port
port = 8080

[search]
# Global policy in case of long queries: "deny" or "warn_only"
long_query_policy = "deny"
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
