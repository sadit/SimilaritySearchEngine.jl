# Events of a project (PLAN.md §5.15). Included after `embedded.jl`, which defines the handle
# that holds the observers, and before `followers.jl`, whose `Follower` is an observer.
#
# An event is a change of a project that a consumer can need to act on. The engine reports it
# once per call that makes the change, after the change is saved, with the range of the `_id`s
# it affects and nothing else. One call is one event, whatever the number of items, and a
# project without observers pays one check of an empty list.

export ProjectEvent, ItemsIndexed, ProjectObserver, on_event, observe!, unobserve!, observers

"""
    ProjectEvent

A change of a project that an observer can act on (PLAN.md §5.15). The engine reports each
event once, after the change is saved, to the observers that [`observe!`](@ref) added. The first
version has one kind, [`ItemsIndexed`](@ref).
"""
abstract type ProjectEvent end

"""
    ItemsIndexed(sp, ep)

The items with `_id` from `sp` to `ep` became searchable. [`index!`](@ref index!(::EmbeddedEngine))
reports the items it indexed. [`append_items!`](@ref) reports the items it appended when the
project indexes on insertion: an exact dense project of `Float32` vectors, or a sparse project.

Each `_id` is reported once by a handle, in increasing order, and the event carries no items: an
observer reads the ones it needs by `_id`, with [`fetch_items`](@ref).
"""
struct ItemsIndexed <: ProjectEvent
    sp::Int
    ep::Int
end

"""
    ProjectObserver

The supertype of the observers of a project (PLAN.md §5.15). A subtype adds methods of
[`on_event`](@ref) for the events it needs, and [`observe!`](@ref) adds an instance to a handle.
A [`Follower`](@ref) is an observer of its leader.
"""
abstract type ProjectObserver end

"""
    on_event(o::ProjectObserver, leader::EmbeddedEngine, e::ProjectEvent)

Called once for each event `e` of `leader`, after the change is saved, in the task that called
the engine, and in the order in which the observers were added. The default method does
nothing, so an observer defines only the methods for the events it needs.

An error of `on_event` does not fail the call of the leader, whose change is saved: the engine
writes it to the log, with the observer and the event. An observer with slow work records what
is pending in `on_event`, and another task does the work. `on_event` must not wait for another
task that changes `leader`: the events of `leader` are reported one at a time.
"""
on_event(::ProjectObserver, ::EmbeddedEngine, ::ProjectEvent) = nothing

"""
    observe!(h::EmbeddedEngine, o::ProjectObserver) -> o

Adds `o` to the observers of `h`, after the ones already added. An observer that `h` already has
is not added again. Events are not saved: `o` receives the events of `h` from this call until
[`unobserve!`](@ref) or [`close_project!`](@ref).
"""
function observe!(h::EmbeddedEngine, o::ProjectObserver)
    _check_open(h)
    lock(h.events_lock) do
        any(x -> x === o, h.observers) || push!(h.observers, o)
    end
    return o
end

"""
    unobserve!(h::EmbeddedEngine, o::ProjectObserver) -> h

Removes `o` from the observers of `h`. Does nothing when `h` does not have it.
"""
function unobserve!(h::EmbeddedEngine, o::ProjectObserver)
    lock(h.events_lock) do
        filter!(x -> x !== o, h.observers)
    end
    return h
end

"""
    observers(h::EmbeddedEngine) -> Vector{ProjectObserver}

The observers of `h`, in the order in which they receive the events. The open followers of `h`
are among them.
"""
observers(h::EmbeddedEngine) = lock(() -> ProjectObserver[o for o in h.observers], h.events_lock)

# Reports to the observers of `handle` the items indexed since the last report, once. Called at
# the end of the calls that index items, after their changes are saved. The counter moves even
# when there are no observers, so that an observer added later receives only later items.
function _report_indexed!(handle::EmbeddedEngine)
    lock(handle.events_lock) do
        n = IndexEngine.indexed_length(handle.engine)
        sp = handle.reported[] + 1
        n >= sp || return nothing
        handle.reported[] = n
        isempty(handle.observers) || _dispatch!(handle, ItemsIndexed(sp, n))
        return nothing
    end
end

function _dispatch!(handle::EmbeddedEngine, e::ProjectEvent)
    # A copy: an observer may remove itself, or another one, while it handles the event.
    for o in copy(handle.observers)
        try
            on_event(o, handle, e)
        catch err
            err isa InterruptException && rethrow()
            @error "the observer $(_observer_name(o)) of the project $(handle.dataset) failed on $e; the change of the project is saved" exception=(err, catch_backtrace())
        end
    end
    return nothing
end

_observer_name(o) = repr(o)
