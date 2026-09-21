#!/usr/bin/env julia

# Text-only inspection of the provisional numerical reference. This helper does
# not include/evaluate that source, read datasets, construct an MPS, or compress.

const DEFAULT_SOURCE = normpath(joinpath(@__DIR__, "..", "Julia", "0mps_calc.jl"))
const SOURCE_PATH = length(ARGS) == 0 ? DEFAULT_SOURCE : length(ARGS) == 1 ? abspath(ARGS[1]) : error("usage: inspect_mps_reference_contract.jl [path-to-0mps_calc.jl]")

isfile(SOURCE_PATH) || error("source file not found: $(SOURCE_PATH)")
const SOURCE_LINES = readlines(SOURCE_PATH)

function find_line(literal::AbstractString; start::Int=1)
    index = findnext(line -> occursin(literal, line), SOURCE_LINES, start)
    isnothing(index) && error("literal not found in $(SOURCE_PATH): $(repr(literal))")
    return index
end

function print_numbered_range(first_line::Int, last_line::Int)
    for line_number in first_line:last_line
        println(lpad(line_number, 5), " | ", SOURCE_LINES[line_number])
    end
end

function print_matching_lines(title::AbstractString, literals)
    println("\n== ", title, " ==")
    found = false
    for (line_number, line) in enumerate(SOURCE_LINES)
        if any(literal -> occursin(literal, line), literals)
            println(lpad(line_number, 5), " | ", line)
            found = true
        end
    end
    found || error("no lines found for section: $(title)")
end

println("source: ", SOURCE_PATH)
println("mode: source text inspection only; no source evaluation and no dataset access")

ordering_start = find_line("# --- helpers ---")
ordering_end = find_line("function mps_truncate"; start=ordering_start) - 1
println("\n== Literal ordering and permutation definitions ==")
print_numbered_range(ordering_start, ordering_end)

print_matching_lines(
    "Literal MPS constructor calls and keyword arguments",
    ("ψ = MPS(",),
)
print_matching_lines(
    "Literal active compression call sites and keyword arguments",
    ("u_rec, chi_c = mps_truncate(u, χ=chi_val", "u_rec, chi_c = mps_truncate(u, cutoff=cf_val"),
)
