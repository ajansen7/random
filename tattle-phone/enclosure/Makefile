# Build printable STLs, the KiCad board outline (DXF) and preview renders.
#   make            -> stl/*.stl + pcb_outline.dxf
#   make renders    -> renders/*.png (needs a display, or xvfb-run on headless Linux)
OPENSCAD ?= openscad
SCAD     := tattle_phone.scad
PARTS    := front back keypad call_cap speaker_cup status_lens
STLS     := $(PARTS:%=stl/%.stl)
XVFB     := $(shell command -v xvfb-run >/dev/null && echo xvfb-run -a)
RENDER    = $(XVFB) $(OPENSCAD) --imgsize=1400,1050 --colorscheme=Tomorrow

all: layout.scad $(STLS) pcb_outline.dxf

layout.scad: ../hardware/layout.py
	python3 ../hardware/layout.py

stl/%.stl: $(SCAD) layout.scad
	@mkdir -p stl
	$(OPENSCAD) -o $@ -D 'part="$*"' -D print_orient=true $(SCAD)

pcb_outline.dxf: $(SCAD) layout.scad
	$(OPENSCAD) -o $@ -D 'part="pcb"' $(SCAD)

renders: layout.scad
	@mkdir -p renders
	$(RENDER) --render -o renders/closed.png --camera=170,-130,260,0,85,12   -D 'part="closed"'   $(SCAD)
	$(RENDER) -o renders/xray.png     --camera=170,-130,260,0,85,12   -D 'part="assembly"' $(SCAD)
	$(RENDER) -o renders/exploded.png --camera=260,-170,190,0,85,28   -D 'part="exploded"' $(SCAD)
	$(RENDER) -o renders/keypad.png   --camera=90,-20,140,0,75,26     -D 'part="keypad"'   $(SCAD)

clean:
	rm -rf stl renders pcb_outline.dxf

.PHONY: all renders clean
