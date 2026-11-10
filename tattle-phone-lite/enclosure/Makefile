# Build printable STLs, the KiCad board outline (DXF) and preview renders.
#   make            -> stl/*.stl + pcb_outline.dxf
#   make renders    -> renders/*.png
OPENSCAD ?= $(shell command -v openscad || echo /Applications/OpenSCAD-2021.01.app/Contents/MacOS/OpenSCAD)
SCAD     := tattle_lite.scad
PARTS    := front back door cap_tattle cap_call cap_hangup speaker_cup
STLS     := $(PARTS:%=stl/%.stl)
XVFB     := $(shell command -v xvfb-run >/dev/null && echo xvfb-run -a)
RENDER    = $(XVFB) "$(OPENSCAD)" --imgsize=1400,1050 --colorscheme=Tomorrow

all: layout.scad $(STLS) pcb_outline.dxf

layout.scad: ../hardware/layout.py
	python3 ../hardware/layout.py

stl/%.stl: $(SCAD) layout.scad
	@mkdir -p stl
	"$(OPENSCAD)" -o $@ -D 'part="$*"' -D print_orient=true $(SCAD)

pcb_outline.dxf: $(SCAD) layout.scad
	"$(OPENSCAD)" -o $@ -D 'part="pcb"' $(SCAD)

# Preview renders (F5 style) keep the colours; OpenSCAD 2021 --render drops them.
renders: layout.scad
	@mkdir -p renders
	$(RENDER) -o renders/closed.png    --camera=170,-140,250,0,85,14  -D 'part="closed"'    $(SCAD)
	$(RENDER) -o renders/back.png      --camera=170,-140,250,0,85,14  -D 'part="back_view"' $(SCAD)
	$(RENDER) -o renders/xray.png      --camera=170,-140,250,0,85,14  -D 'part="assembly"'  $(SCAD)
	$(RENDER) -o renders/exploded.png  --camera=260,-180,170,0,85,20  -D 'part="exploded"'  $(SCAD)

clean:
	rm -rf stl renders pcb_outline.dxf

.PHONY: all renders clean
