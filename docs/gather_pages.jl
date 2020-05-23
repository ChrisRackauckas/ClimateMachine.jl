
"""
    replace_reverse(x, p...; count::Union{Integer,Nothing}=nothing)

`replace` method, but `count` is applied in reverse.
"""
function replace_reverse(x, p...; count::Union{Integer,Nothing}=nothing)
  from = getproperty.(p, :first)
  to = getproperty.(p, :second)
  reversed_pairs = Pair.(reverse.(from), reverse.(to))
  x_reversed = reverse(x)
  if count==nothing
    return String(collect(Iterators.reverse(replace(x_reversed, reversed_pairs...))))
  else
    return String(collect(Iterators.reverse(replace(x_reversed, reversed_pairs...; count=count))))
  end
end

if Sys.isunix()
    const path_separator = '/'
elseif Sys.iswindows()
    const path_separator = '\\'
else
    error("path_separator for this OS need to be defined")
end

"""
    transform_page(filename)

Transform filename to title
seen in the navigation panel.
"""
function transform_page(filename)
  titlename = String(first(splitext(filename)))
  titlename = replace(titlename, "_" => " ")
  titlename = titlecase(titlename)
  return titlename
end

"""
    transform_subpage(local_path)

Transform local path to sub-page
title, seen in the navigation panel.
"""
transform_subpage(local_path) =
    join(split_by_camel_case(local_path), " ")

"""
    split_by_camel_case(s::AbstractString)

Splits a string into an array of strings,
by its `CamelCase`. Examples:

```julia
julia> split_by_camel_case("CamelCase")
2-element Array{SubString{String},1}:
 "Camel"
 "Case"
julia> split_by_camel_case("PDESolver")
2-element Array{SubString{String},1}:
 "PDE"
 "Solver"
```
"""
split_by_camel_case(obj::AbstractString) =
    split(obj, r"((?<=\p{Ll})(?=\p{Lu})|(?<=\p{Lu})(?=\p{Lu}[^\p{Lu}]+))")

"""
    gather_pages!(
        array::Array,
        filename::S,
        folders_with_only_files::Vector{S},
        transform_page::Function,
        transform_subpage::Function,
        fullpath = dirname(filename),
    ) where {S<:AbstractString}

Construct a nested array of `Pair`s whose
keys mirror the given folder structure and
values which point to the files.
"""
function gather_pages!(
    array::Array,
    filename::S,
    folders_with_only_files::Vector{S},
    transform_page::Function,
    transform_subpage::Function,
    fullpath = dirname(filename),
) where {S<:AbstractString}
    paths = splitpath(filename)
    if length(paths) == 1
        file = paths[1]
        push!(array, Pair(String(transform_page(file)), String(joinpath(fullpath, file))))
    else
        key = transform_subpage(paths[1])
        if !any(x.first == key for x in array)
            push!(array, Pair(key, Any[]))
        end
        gather_pages!(array[end].second,
                      joinpath(paths[2:end]...),
                      folders_with_only_files,
                      transform_page,
                      transform_subpage,
                      fullpath)
    end
end

"""
    gather_pages(;
        filenames::Union{Array,Nothing}=nothing,
        transform_page::Function=transform_page,
        transform_subpage::Function=transform_subpage,
        transform_extension=x->x,
        transform_path=x->x,
    )

Construct a nested array of `Pair`s
given an array of files.

 - `filenames` Array of files (using relative paths)
 - `transform_page` function to transform the page (title) in navigation panel
 - `transform_subpage` function to transform the sub-page in navigation panel
 - `transform_extension` transform extensions (e.g., ".jl" to ".md")
                         a useful one is `replace_reverse(x, ".jl" => ".md"; count=1)`
 - `transform_path` transform path
"""
function gather_pages(;
    filenames::Union{Array,Nothing}=nothing,
    transform_page::Function=transform_page,
    transform_subpage::Function=transform_subpage,
    transform_extension=x->x,
    transform_path=x->x,
    )

    filenames = map(x -> transform_extension(x), filenames)
    filter!(x -> !(x == path_separator), filenames)
    filenames = map(x -> String(lstrip(x, path_separator)), filenames)
    filenames = map(x -> transform_path(x), filenames)
    dirnames = collect(Set(dirname.(filenames)))

    dirnames = [x for x in dirnames if !any(occursin(x, y) && !(x == y) for y in dirnames)]
    folders_with_only_files = basename.(dirnames)
    array = Any[]
    for file in filenames
        gather_pages!(array,
                      file,
                      folders_with_only_files,
                      transform_page,
                      transform_subpage)
    end
    array = array[1].second

    return array, filenames
end
