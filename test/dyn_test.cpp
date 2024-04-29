#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <string>
#include <sstream>
#include <vector>
#include <queue>
#include <map>
#include <chrono>
#include <stdexcept>
#include "parse_points.hpp"
#include "graph/adj.hpp"
#include "algo/HNSW.hpp"
#include "dist.hpp"
#include "parlay.hpp"
#include "benchUtils.h"
#include "cpam.hpp"
using ANN::HNSW;

parlay::sequence<size_t> per_visited;
parlay::sequence<size_t> per_eval;
parlay::sequence<size_t> per_size_C;

namespace ANN::external{

auto def_custom_tag()
{
	return custom_tag_parlay{};
}

} // namespace ANN::external

template<typename T>
point_converter_default<T> to_point;

template<typename T>
class gt_converter{
public:
	using type = parlay::sequence<T>;
	template<typename Iter>
	type operator()([[maybe_unused]] uint32_t id, Iter begin, Iter end)
	{
		using type_src = typename std::iterator_traits<Iter>::value_type;
		static_assert(std::is_convertible_v<type_src,T>, "Cannot convert to the target type");

		const uint32_t n = std::distance(begin, end);

		// T *gt = new T[n];
		auto gt = parlay::sequence<T>(n);
		for(uint32_t i=0; i<n; ++i)
			gt[i] = *(begin+i);
		return gt;
	}
};

template<class DescLegacy>
struct desc{
	using point_t = point<typename DescLegacy::type_elem>;
	using coord_t = typename point_t::coord_t;
	using dist_t = float;
	static dist_t distance(const coord_t &cu, const coord_t &cv, uint32_t dim){
		return DescLegacy::distance(cu, cv, dim);
	}

	template<typename Nid, class Ext>
	using graph_t = ANN::graph::adj_seq<Nid,Ext>;

	template<typename Nid, class Ext>
	using graph_aux = ANN::graph::adj_map<Nid,Ext>;
};

template<class DescLegacy>
struct desc_cpam: desc<DescLegacy>{
	template<typename Nid, class Ext, class Edge=Nid>
	using graph_t = graph_cpam<Nid,Ext,Edge>;

	template<typename Nid, class Ext, class Edge=Nid>
	using graph_aux = graph_cpam<Nid,Ext,Edge>;
};


// Visit all the vectors in the given 2D array of points
// This triggers the page fetching if the vectors are mmap-ed
template<class T>
void visit_point(const T &array, size_t dim0, size_t dim1)
{
	parlay::parallel_for(0, dim0, [&](size_t i){
		const auto &a = array[i];
		[[maybe_unused]] volatile auto elem = a.get_coord()[0];
		for(size_t j=1; j<dim1; ++j)
			elem = a.get_coord()[j];
	});
}

template<class G>
void print_stat(const G &g)
{
	const uint32_t height = g.get_height();
	printf("Highest level: %u\n", height);
	puts("level     #vertices         edges  avg. deg");
	for(uint32_t i=0; i<=height; ++i)
	{
		const uint32_t level = height-i;
		size_t cnt_vertex = g.num_nodes(level);
		size_t cnt_degree = g.num_edges(level);
		printf("#%2u: %14lu %16lu %10.2f\n", 
			level, cnt_vertex, cnt_degree, float(cnt_degree)/cnt_vertex
		);
	}
	
}

template<class G, class Seq>
auto find_nbhs(const G &g, const Seq &q, uint32_t k, uint32_t ef)
{
	const size_t cnt_query = q.size();
	per_visited.resize(cnt_query);
	per_eval.resize(cnt_query);
	per_size_C.resize(cnt_query);

	using seq_result = parlay::sequence<typename G::result_t>;
	parlay::sequence<seq_result> res(cnt_query);
	auto search = [&]{
		parlay::parallel_for(0, cnt_query, [&](size_t i){
			ANN::algo::search_control ctrl{};
			ctrl.log_per_stat = i;
			// ctrl.beta = beta;
			res[i] = g.template search<seq_result>(q[i].get_coord(), k, ef, ctrl);
		});
	};

	puts("Warmup");
	search();

	parlay::internal::timer t;
	const uint32_t rounds = 3;
	for(uint32_t i=0; i<rounds; ++i)
		search();
	const double time_query = t.next_time()/rounds;
	const double qps = cnt_query/time_query;
	printf("Find neighbors: %.4f s, %e kqps\n", time_query, qps/1000);

	printf("# visited: %lu\n", parlay::reduce(per_visited,parlay::addm<size_t>{}));
	printf("# eval: %lu\n", parlay::reduce(per_eval,parlay::addm<size_t>{}));
	printf("size of C: %lu\n", parlay::reduce(per_size_C,parlay::addm<size_t>{}));

	parlay::sort_inplace(per_visited);
	parlay::sort_inplace(per_eval);
	parlay::sort_inplace(per_size_C);
	const double tail_ratio[] = {0.9, 0.99, 0.999};
	for(size_t i=0; i<sizeof(tail_ratio)/sizeof(*tail_ratio); ++i)
	{
		const auto r = tail_ratio[i];
		const uint32_t tail_index = r*cnt_query;
		printf("%.4f tail stat (at %u):\n", r, tail_index);

		printf("\t# visited: %lu\n", per_visited[tail_index]);
		printf("\t# eval: %lu\n", per_eval[tail_index]);
		printf("\tsize of C: %lu\n", per_size_C[tail_index]);
	}

	return res;
}

template<class U, class Seq, class Point>
auto CalculateOneKnn(const Seq &data, const Point &q, uint32_t dim, uint32_t k)
{
	static_assert(std::is_same_v<Point, typename U::point_t>);

	using pid_t = typename U::point_t::id_t;
	std::priority_queue<std::pair<float, pid_t>> top_candidates;
	float lower_bound = std::numeric_limits<float>::min();
	for(size_t i=0; i<data.size(); ++i)
	{
		const auto &u = data[i];
		float dist = U::distance(u.get_coord(), q.get_coord(), dim);

		// only keep the top k
		if (top_candidates.size() < k || dist < lower_bound) {
			top_candidates.emplace(dist, u.get_id());
			if (top_candidates.size() > k)
				top_candidates.pop();
			lower_bound = top_candidates.top().first;
		}
	}

	parlay::sequence<pid_t> knn;
	while (!top_candidates.empty()) {
		knn.emplace_back(top_candidates.top().second);
		top_candidates.pop();
	}
	std::reverse(knn.begin(), knn.end());
	return knn;
}

template<class U, class S1, class S2>
auto ConstructKnng(const S1 &data, const S2 &qs, uint32_t dim, uint32_t k)
{
	using pid_t = typename U::point_t::id_t;
	parlay::sequence<parlay::sequence<pid_t>> res(qs.size());
	parlay::parallel_for(0, qs.size(), [&](size_t i){
		res[i] = CalculateOneKnn<U>(data, qs[i], dim, k);
	});
	return res;
}

template<class S1, class S2, class S3>
void calc_recall(const S1 &q, const S2 &res, const S3 &gt, uint32_t k)
{
	const size_t cnt_query = q.size();
//	uint32_t cnt_all_shot = 0;
	std::vector<uint32_t> result(k+1);
	printf("measure recall@%u on %lu queries\n", k, cnt_query);
	for(uint32_t i=0; i<cnt_query; ++i)
	{
		uint32_t cnt_shot = 0;
		for(uint32_t j=0; j<k; ++j)
			if(std::find_if(res[i].begin(),res[i].end(),[&](const auto &p){
				return p.pid==gt[i][j];}) != res[i].end()) // TODO: fix naming
			{
				cnt_shot++;
			}
		result[cnt_shot]++;
	}
	size_t total_shot = 0;
	for(size_t i=0; i<=k; ++i)
	{
		printf("%u ", result[i]);
		total_shot += result[i]*i;
	}
	putchar('\n');
	printf("recall: %.6f\n", float(total_shot)/cnt_query/k);
}

template<typename U>
void run_test(commandLine parameter) // intend to be pass-by-value manner
{
	const char *file_in = parameter.getOptionValue("-in");
	const size_t size_init = parameter.getOptionLongValue("-init", 0);
	const size_t size_step = parameter.getOptionLongValue("-step", 0);
	size_t size_max = parameter.getOptionLongValue("-max", 0);
	const float m_l = parameter.getOptionDoubleValue("-ml", 0.36);
	const uint32_t m = parameter.getOptionIntValue("-m", 40);
	const uint32_t efc = parameter.getOptionIntValue("-efc", 60);
	const float alpha = parameter.getOptionDoubleValue("-alpha", 1);
	const float batch_base = parameter.getOptionDoubleValue("-b", 2);
	const char* file_query = parameter.getOptionValue("-q");
	const uint32_t k = parameter.getOptionIntValue("-k", 10);
	const uint32_t ef = parameter.getOptionIntValue("-ef", m*20);
	
	parlay::internal::timer t("run_test:prepare", true);

	using T = typename U::point_t::elem_t;
	auto [ps,dim] = load_point(file_in, to_point<T>, size_max);
	t.next("Load the base set");
	printf("%s: [%lu,%u]\n", file_in, ps.size(), dim);

	if(ps.size()<size_max)
	{
		size_max = ps.size();
		printf("size_max is corrected to %lu\n", size_max);
	}

	auto [q,_] = load_point(file_query, to_point<T>);
	t.next("Load queries");
	printf("%s: [%lu,%u]\n", file_query, q.size(), _);

	visit_point(ps, size_max, dim);
	visit_point(q, q.size(), dim);
	t.next("Prefetch vectors");

	HNSW<U> g(dim, m_l, m, efc, alpha);
	std::vector<HNSW<U>> snapshots;
	puts("Initialize HNSW");

	for(size_t size_last=0, size_curr=size_init;
		size_curr<=size_max;
		size_last=size_curr, size_curr+=size_step)
	{
		printf("Increasing size from %lu to %lu\n", size_last, size_curr);

		puts("Insert points");
		parlay::internal::timer t("run_test:insert", true);
		auto ins_begin = ps.begin()+size_last;
		auto ins_end = ps.begin()+size_curr;
		g.insert(ins_begin, ins_end, batch_base);
		t.next("Finish insertion");

		snapshots.push_back(g);

		puts("Collect statistics");
		print_stat(g);

		puts("Search for neighbors");
		auto res = find_nbhs(g, q, k, ef);

		puts("Generate groundtruth");
		auto baseset = parlay::make_slice(ps.begin(), ins_end);
		auto gt = ConstructKnng<U>(baseset, q, dim, k);

		puts("Compute recall");
		calc_recall(q, res, gt, k);

		puts("---");
	}
}

int main(int argc, char **argv)
{
	for(int i=0; i<argc; ++i)
		printf("%s ", argv[i]);
	putchar('\n');

	commandLine parameter(argc, argv, 
		"-type <elemType> -dist <distance>"
		"-ml <m_l> -m <m> -efc <ef_construction> -alpha <alpha> "
		"-in <baseset> -q <queries> "
		"-init <init_size> -step <step_size> -max <max_size>"
		"-k <recall@k> -ef <ef_query> [-beta <beta>,...]"
	);

	const char *dist_func = parameter.getOptionValue("-dist");
	auto run_test_helper = [&](auto type){ // emulate a generic lambda in C++20
		using T = decltype(type);
		if(!strcmp(dist_func,"L2"))
			run_test<desc_cpam<descr_l2<T>>>(parameter);
		/*
		else if(!strcmp(dist_func,"angular"))
			run_test<desc<descr_ang<T>>>(parameter);
		else if(!strcmp(dist_func,"ndot"))
			run_test<desc<descr_ndot<T>>>(parameter);
		*/
		else throw std::invalid_argument("Unsupported distance type");
	};

	const char* type = parameter.getOptionValue("-type");
	if(!strcmp(type,"uint8"))
		run_test_helper(uint8_t{});
	/*
	else if(!strcmp(type,"int8"))
		run_test_helper(int8_t{});
	else if(!strcmp(type,"float"))
		run_test_helper(float{});
	*/
	else throw std::invalid_argument("Unsupported element type");
	return 0;
}
