using Literate
Literate.markdown("nextjournal.jl", "."; documenter=false)
Literate.notebook("nextjournal.jl", ".")
