# WaveFunctionCollapse with quantum wave function collapse (QWFC)

![example maps](png "example maps")

QWFC is an experimental implementation of the [WFC](https://github.com/mxgmn/WaveFunctionCollapse) algorithm for quantum
computers. The original program generates images that appear similar to an input image by randomly drawing image
elements in an iterative fashion. A short description of the algorithm can be found
on https://github.com/mxgmn/WaveFunctionCollapse, a more in-depth description is for example
presented [here](https://robertheaton.com/2018/12/17/wavefunction-collapse-algorithm.

WFC is a quantum-inspired but in fact completely classical algorithm: "Unobserved" parts of an image are in a "
superposition" of various possibilities depending on the already "collapsed" (i.e., selected) parts. QWFC is an attempt
to implement the quantum concept from which WFC is inspired on an actual gate-based quantum computer by creating a
quantum circuit (with [Qiskit](https://qiskit.org/)) that prepares a suitable superposition state. To this end, the tilemap approach of WFC is considered, in
which each image — or map — is composed of a set of tiles and certain rules restrict the placement of these tiles
depending on adjacent tiles. Each tile is encoded in or or more qubits such that the prepared state is a superposition
of all possible tile layouts. A measurement consequently reveals a random realization. QWFC is not a one-to-one
recreation of WFC, but a simlified version that has the goal to achieve a similar end result.

## Algorithm

**Input:**

1. Definition of the tileset: Alphabet of tiles.
    * Each tile is encoded by a consecutive integer.
    * `N` qubits are required to encode one tile with an alphabet of size `2^N`.
2. Definition of the map: Ordered set of tiles with adjacency relations.
    * In total, `N*k` qubits are required to encode an alphabet of size `2^N` and `k` tiles on the map.
3. Defintion of the generator: Tile placement probabilities based on adjacency relations.

**Circuit generation:**

1. Start with the first tile in the list.
2. Prepare the qubits in a superposition state based on already traversed tiles and the specified placement
   probabilities.
3. Repeat from 2. with the next tile in the list until all qubits have been prepared.

## Example

As a trivial example, consider a one-dimensional map with only white/black tiles that is traversed from left to right
with the following checkerboard rules:

1. If the tile on the left is black, only allow a white tile (with 100% probability).
2. If the tile on the left is white, only allow a black tile (with 100% probability).
3. If there is no tile on the left, choose either a black or white tile (each with 50% probability).
   This map can be encoded with one qubit per tile: state `0` corresponds to white and state `1` corresponds to black.

For the first tile, rule 3 applies. It is therefore prepared in a superposition state with a simple Hadamard gate. For
the second tile, rule 1 or 2 apply. These rules can be implemented with a controlled rotation. Similarly for subsequent
tiles.

Consider a very simple map consisting of three tiles (reed - green - blue):

![checkerboard tiles](png "checkerboard tiles")

The resulting circuit to produce checkerboard tiles is then:

![checkerboard circuit](png "checkerboard circuit")

Measuring this circuit reveals two tile layouts (starting with a white tile or starting with a black tile) with the same
probability:

![checkerboard layout 0](png "checkerboard layout A")
![checkerboard layout 1](png "checkerboard layout B")

That is, the circuit produces a state that represents *all* valid layout combinations in a superposition (two in this
simple case). Its measurement *collapses* the state onto one definite layout. That's the simple idea behind QWFC.

## Hybrid algorithm

Since current quantum hardware is very limited, a hybrid quantum-classical algorithm is also implemented in which the
map is traversed with a sliding window. For each iteration, a circuit is generated and measured for the sliding window
region (also taking the adjacency relations outside of the window into account). With this approach, significantly
bigger maps can be generated.

![hybrid algorithm](png "hybrid")