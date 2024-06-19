# ANNlib

ANNlib is a library providing highly-optimized ANN algorithms and commonly-used data structures as the building blocks to accelerate development of your own ANN system.

It provides highly reusable and configurable components at different levels, including out-of-the-box ANN classes, algorithm templates, interfaces of graph structures, helper functions, adapters , I/O utilities, and more.

These components are presented as header-only C++ source code and have no needs of compilation nor installation. To include the header files and start up your first ANN program with ANNlib, see the section [Getting started](#getting-started) below. 

ANNlib is under an active development and keeps adding more functions. With ANNlib, one can easily set up a usable base case of a ANN program and quickly start either relevant research or specific improvements with it.



## Core Design

ANNlib separates a full ANN system into three dimensions: algorithm, data structure, and ID management, and provides decoupled and replaceable building blocks for each. It is thus possible to develop each component individually and easy to add own code compatible with the library.

Algorithm is the core logic component that essentially defines how to build and search through an ANN index. Algorithms in ANNlib are presented as C++ templates accepting different types of other components designated by the user. Such design makes algorithms decoupled from other components and only interact with them through standard interfaces. The user can safely modify an algorithm with no need of collateral changes to data structures or ID management.

Data structures are highly-optimized application-specific data types that serve the storage of ANN index. With the out-of-the-box data structures, the users are able to focus on developing their own features rather than take care of the storage details. ANNlib provides carefully-designed interfaces to uniformly access data structures for storage, that both consider the semantic representability, the ease of use, and the runtime efficiency. These data structures provide a degree of freedom to store ANN indices in anywhere (e.g., memory, disks, distributed locations, or mixture of multiple medias) and in any form (e.g., adjacent lists, tree embeddings, or concurrent data structures).

ID management provides a bridge that connects, of each point, the identity assigned by the user and the internal representation used for algorithmic logics and data storage. In the user's point of view, all he can handle is a user-defined identifier associated with each point, and he operate (update/query) on the ANN system by identifier. On the other hand, from the perspective of an ANN system, it needs a value of independent type to refer to the internal object for each point. ID management provides a mapping between these two, and can be further used to reduce complexity in distributed scenarios.

Besides, ANNlib is designed not to depend on any third-party libraries while allows users to customize infrastructure, such as the parallel framework and memory allocator, by a few lines of code. This design makes industrial users easily adapt ANNlib to their own products without modifying any library code and replace frequently-used functions for better performance in clear form.



## Main Components

Up to the latest release, `ANNlib` primarily provides:

- re-implementations of algorithms `vamana`, `HNSW`,  and `HCNNG` under the `algo/` folder
- data structures `adj_seq` and `adj_map` as generic adjacent-list-based variants for storing graph-based ANN indices under the `graph/` folder, and `graph_cpam` as an example of adaption to the `PAM/CPAM` graphs at `test/cpam.hpp`
- basic ID management classes `trivial` and `direct` under the `map/` folder
- example code of customizing the infrastructure, at `custom/undef.hpp` as the default settings, and at `test/parlay.hpp` to integrate with the [ParlayLib](https://github.com/cmuparlay/parlaylib).

Besides the above class templates corresponding to the core designs, ANNlib also provides

- common algorithmic functions `beamSearch` and `prune_heuristic` in `algo/algo.hpp` which are deeply optimized and highly configurable.
- helper functions and classes and low-level components in the `util/` folder
- I/O utilities in the `io/` folder
- examples of how to use ANNlib and fast develop own applications, plus some benchmarks and tests, in the `test/` folder

See [Tutorial](docs/tutorial.md) for more usage of the ANNlib.



## Requirements

ANNlib is written in standard C++ and should work with any C++20-compliant compilers. It has been well test with GCC 13.2.0.

The support to C++17 has been ended, due to a great need of `concept` and `std::ranges` for new features, and the legacy code in C++17 is moved into the `C++17` branch with seldom maintenance and backports but accepting merge requests.

If you run into any problem, please open an issue and leave the comments.



## Getting started

Beginning the initial trial with ANNlib, we first define a `Point` class and create a few base points.

```c++
#include <vector>
using std::vector;

struct Point{
    const char *name;
    vector<float> coord;
};

int main(){
    vector<Point> ps{
        {"left", {0,0}},
        {"right", {4,0}},
        {"top", {2,1}}
    };
    return 0;
}
```

Here, `Points` consists of two fields, `name` and `coord`, that indicate the unique name and the coordinate of a point, respectively. We create three points named `left`, `right`, and `top` with corresponding coordinates `(0,0)`, `(4,0)`, and `(2,1)`, and store them in a vector `ps`. You might have noticed that both `Point` and its fields are user-defined types; therefore, to make the ANNlib aware of these types and construct an index from the points, we need to define some extra types and functions following the certain conventions.

```cpp
struct Point{
    const char *name;
    vector<float> coord;

    /* define types and funcs for ANNlib to use */
    using id_t = const char*;
    using elem_t = float;
    using coord_t = vector<float>;

    id_t get_id() const {return name;}
    const coord_t& get_coord() const {return coord;}
};
```

In total three types and two member functions are required in `Point` for the ANNlib to use. 

`id_t` is the type of identifier, a unique value that ANNlib uses to distinguish the points and reports back from the search; here, we use the point's name as its identifier, whereas you can use any other values that fit your application such as a numeric key as long as it is distinct over the base points.

`coord_t` is the type of coordinates, and `elem_t` is the type of each single " number" in a coordinate -- the numbers have not to be numeric; instead, they can be of any types that meet the [`elem_t` requirements](#).

`get_id()` is a const-qualified member function that returns the identifier of this point; similarly, `get_coord()` returns the coordinate. For more details about the interfaces of `get_id` and `get_coord`, see [`point_t` requirements](#)

Aside from additional members of `Point`, the class itself has to be exposed in a descriptor class along with other necessary fields.

```cpp
struct Desc{
    using point_t = Point;

    using dist_t = float;
    static dist_t distance(const auto &cu, const auto &cv, uint32_t dim){
        // user-defined distance function: 2d L2 norm
        assert(dim==2);
        auto d0=cu[0]-cv[0], d1=cu[1]-cv[1];
        return d0*d0 + d1*d1;
    }

    template<typename Nid, class Ext, class Edge>
    using graph_t = typename ANN::graph::adj_map<Nid,Ext,Edge>;
};
```

Here, we define a descriptor class `Desc` for passing configurations to a graph-based ANN that will be soon shown in the next step. We make ANNlib aware of the previously defined point type by assigning it to `point_t`. We also need to define the distance function and its return type `dist_t` necessarily. As we are going to use a graph-based ANN, the field of `graph_t` is required to indicate the data structure to store the graph, which can be either the provided components in the ANNlib or any user-defined type following the [interface requirements](#interfaces). A common choice is to store the graph index in a map-of-vector structure, which is provided as `ANN::graph::adj_map`, where edges are arranges as a vector and keyed by their incident vertex in a map.

Piecing the things together, we have the code to try out the ANNlib by inserting points in `ps` and asking for the two nearest neighbors of the position `(3,3)`. Note the constructor of `ANN::vamana` accepts one type template parameter as the descriptor, and one non-optional parameter as the dimension of input points. A few default parameters such as beam size are omitted in this example, and you can find the full parameter list in [vamana APIs](#). The `search` method accepts a coordinate of `Point:coord_t` type as the search target, a $k$ for $k$-NN as the number of neighbors to return, and a beam size greater than or equal to $k$ that trades off the search time and quality. The `search ` method also accept a type template parameter as the return type. If not indicated, `std::vector<vamana<Desc>::result_t>` is used by default where `result_t` is essentially a pair of `dist_t` and `Point::id_t` containing the distance to the target and the identifier, respectively, of a returned neighbor.

```cpp
#include <cstdio>
#include <cassert>
#include <vector>
#include "custom/undef.hpp"
#include "graph/adj.hpp"
#include "algo/vamana.hpp"
using std::vector;

struct Point{...};
struct Desc{...};

int main(){
    vector<Point> ps{...};
    ANN::vamana<Desc> index(2); // dimension = 2
    index.insert(ps.begin(), ps.end());
    auto res = index.search(vector<float>{3,3}, 2, 10);
    for(auto [d,u] : res)
        printf("[%s] in distance of %.1f\n", u, d);
    return 0;
}
```

Now we compile and run the above example using GCC for instance, assuming the code is saved as `start.cpp`, and fill `<path_to_ANNlib>` with the actual path to the ANNlib in your computer, which you should probably `git clone` first.

```shell
g++ start.cpp -std=c++20 -I<path_to_ANNlib> -o start
./start
```

The expected output is

```
[top] in distance of 5.0
[right] in distance of 10.0
```

since among the given points `left:(0,0)`, `right:(4,0)`, and `top:(2,1)`,  the points `top` and `right` are the two nearest to the target`(3,3)` under the defined distance function.



## Interfaces

Beyond simply using the provided components in the ANNlib, for ones who want to understand the implementations or develop their own functions, please read the detailed interface documentation in [Interfaces](docs/interfaces.md).
