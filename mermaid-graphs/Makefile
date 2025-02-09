BUILDDIR = ./build

%.png: %.mmd
	mkdir -p $(BUILDDIR)
	mmdc -i $< -o $(BUILDDIR)/$@

clean:
	rm -rf $(BUILDDIR)

build: clean $(patsubst %.mmd,%.png,$(wildcard *.mmd))

.PHONY: build clean
