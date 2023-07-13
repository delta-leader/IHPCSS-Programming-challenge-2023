/**
 * @file main.f08
 * @brief This file provides you with the original implementation of pagerank.
 * Your challenge is to optimise it using OpenMP and/or MPI.
 * @author Ludovic Capelli (l.capelli@epcc.ed.ac.uk)
 **/
 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include <stdint.h>
#include <string.h>

/// The number of vertices in the graph.
#define GRAPH_ORDER 1000
/// Parameters used in pagerank convergence, do not change.
#define DAMPING_FACTOR 0.85
/// The number of seconds to not exceed forthe calculation loop.
#define MAX_TIME 10

/**
 * @brief Indicates which vertices are connected.
 * @details If an edge links vertex A to vertex B, then adjacency_matrix[A][B]
 * will be 1.0. The absence of edge is represented with value 0.0.
 * Redundant edges are still represented with value 1.0.
 */
int8_t adjacency_matrix[GRAPH_ORDER][GRAPH_ORDER];
int crs[GRAPH_ORDER][GRAPH_ORDER];
int crs_cnt[GRAPH_ORDER];
double outdegree[GRAPH_ORDER];
double max_diff = 0.0;
double min_diff = 1.0;
double total_diff = 0.0;
 
void initialize_graph(void)
{
    for(int i = 0; i < GRAPH_ORDER; i++)
    {
        for(int j = 0; j < GRAPH_ORDER; j++)
        {
            adjacency_matrix[i][j] = 0;
        }
    }
}

/**
 * @brief Calculates the pagerank of all vertices in the graph.
 * @param pagerank The array in which store the final pageranks.
 */
void calculate_pagerank(double pagerank[])
{
    double initial_rank = 1.0 / GRAPH_ORDER;
 
    // Initialise all vertices to 1/n.
    for(int i = 0; i < GRAPH_ORDER; i++)
    {
        pagerank[i] = initial_rank;
    }
 
    double damping_value = (1.0 - DAMPING_FACTOR) / GRAPH_ORDER;
    double diff = 1.0;
    size_t iteration = 0;
    double start = omp_get_wtime();
    double elapsed = omp_get_wtime() - start;
    double start_t[5];
    double elapsed_t[5];
    double time_per_iteration = 0;
    double new_pagerank[GRAPH_ORDER];
    for(int i = 0; i < GRAPH_ORDER; i++)
    {
        new_pagerank[i] = 0.0;
    }
    for(int i = 0; i < GRAPH_ORDER; i++)
    {
	    outdegree[i] = 0;
	    for(int j = 0; j < GRAPH_ORDER; j++)
        {
	        if (adjacency_matrix[i][j])
            {
		        outdegree[i]++;
	        }
	    }
	    outdegree[i] = 1/outdegree[i];
    }
    for(int j = 0; j < GRAPH_ORDER; j++)
    {
	    crs_cnt[j] = 0;
        int cnt = 0;
	    for(int i = 0; i < GRAPH_ORDER; i++)
        {
	        if (adjacency_matrix[i][j])
            {
		        crs_cnt[j]++;
                crs[j][cnt++]=i;
	        }
	    }
    }
    // map the data on the gpu
    // If running on a single node, we don't need to transfer any of the arrays back to main memory
    #pragma omp target enter data map(to: pagerank[:GRAPH_ORDER], outdegree[:GRAPH_ORDER], crs[:GRAPH_ORDER*GRAPH_ORDER], crs_cnt[:GRAPH_ORDER]) map(alloc: new_pagerank[:GRAPH_ORDER])
    // If we exceeded the MAX_TIME seconds, we stop. If we typically spend X seconds on an iteration, and we are less than X seconds away from MAX_TIME, we stop.
    while(elapsed < MAX_TIME && (elapsed + time_per_iteration) < MAX_TIME)
    {
        double iteration_start = omp_get_wtime();
        start_t[0] = omp_get_wtime();

        // I pulled this loop out again because it prevents the nested loop from collapsing
        // should run pretty efficient anyway
        //#pragma omp target teams distribute parallel for
        /*
        for(int i = 0; i < GRAPH_ORDER; i++)
        {
            new_pagerank[i] = 0.0;
        }
        */
        memcpy(new_pagerank, (double const[1000]){ 0 }, 1000 * sizeof(double));
        elapsed_t[0] = omp_get_wtime() - start_t[0];
        start_t[1] = omp_get_wtime();
 
        //#pragma omp target data map(from:new_pagerank[:1000])
        #pragma omp single
        {
            #pragma omp target teams distribute nowait
            for(int i = 0; i < 1000; i++)
            {
                double sum = 0.0;
                int t = crs_cnt[i];
                int cnt = 0;
                #pragma omp parallel for simd reduction(+:sum)
                for(int j = 0; j < t; j++)
                {
                    int idx = crs[i][j];
                    sum += pagerank[idx] * outdegree[idx];
                }
                new_pagerank[i] = sum;
            }

            /*
            #pragma omp target teams distribute device(1) nowait
            for(int i = 500; i < 1000; i++)
            {
                double sum = 0.0;
                int t = crs_cnt[i];
                int cnt = 0;
                #pragma omp parallel for simd reduction(+:sum)
                for(int j = 0; j < t; j++)
                {
                    int idx = crs[i][j];
                    sum += pagerank[idx] * outdegree[idx];
                }
                new_pagerank[i] = sum;
            }
            */
        }

        elapsed_t[1] = omp_get_wtime() - start_t[1];
        start_t[2] = omp_get_wtime();

        // pulled this one out again as well
        // we need a reduction on diff
        // and diff is needed on host memory
        diff = 0.0;
        //#pragma omp target teams distribute parallel for reduction(+:diff) map(tofrom:diff)
        for(int i = 0; i < GRAPH_ORDER; i++)
        {
            new_pagerank[i] = DAMPING_FACTOR * new_pagerank[i] + damping_value;
	        diff += fabs(new_pagerank[i] - pagerank[i]);
        }
        
        max_diff = (max_diff < diff) ? diff : max_diff;
        total_diff += diff;
        min_diff = (min_diff > diff) ? diff : min_diff;

        elapsed_t[2] = omp_get_wtime() - start_t[2];
        start_t[3] = omp_get_wtime();
 
        double pagerank_total = 0.0;
        // we need a reduction on pagerank_total
        memcpy(new_pagerank, pagerank, 1000 * sizeof(double));
        //#pragma omp target teams distribute parallel for reduction(+:pagerank_total) map(tofrom:pagerank_total)
        for(int i = 0; i < GRAPH_ORDER; i++)
        {
            //pagerank[i] = new_pagerank[i];
            pagerank_total += pagerank[i];
        }
        if(fabs(pagerank_total - 1.0) >= 1E-12)
        {
            printf("[ERROR] Iteration %zu: sum of all pageranks is not 1 but %.12f.\n", iteration, pagerank_total);
        }
        elapsed_t[3] = omp_get_wtime() - start_t[3];
 
	    double iteration_end = omp_get_wtime();
	    elapsed = omp_get_wtime() - start;
	    iteration++;
	    time_per_iteration = elapsed / iteration;
    }
    
    printf("%zu iterations achieved in %.2f seconds\n", iteration, elapsed);
    printf("1. loop (initialization)  %.12f\n",  elapsed_t[0]/iteration);
    printf("2. loop (page rank)       %.12f\n",  elapsed_t[1]/iteration);
    printf("3. loop (diff)            %.12f\n",  elapsed_t[2]/iteration);
    printf("4. loop (page_rank total) %.12f\n",  elapsed_t[3]/iteration);
}

/**
 * @brief Populates the edges in the graph for testing.
 **/
void generate_nice_graph(void)
{
    printf("Generate a graph for testing purposes (i.e.: a nice and conveniently designed graph :) )\n");
    double start = omp_get_wtime();
    initialize_graph();
    for(int i = 0; i < GRAPH_ORDER; i++)
    {
        for(int j = 0; j < GRAPH_ORDER; j++)
        {
            int source = j;
            int destination = i;
            if(i != j)
            {
                adjacency_matrix[source][destination] = 1;
            }
        }
    }
    printf("%.2f seconds to generate the graph.\n", omp_get_wtime() - start);
}

/**
 * @brief Populates the edges in the graph for the challenge.
 **/
void generate_sneaky_graph(void)
{
    printf("Generate a graph for the challenge (i.e.: a sneaky graph :P )\n");
    double start = omp_get_wtime();
    initialize_graph();
    for(int i = 0; i < GRAPH_ORDER; i++)
    {
        for(int j = 0; j < GRAPH_ORDER - i; j++)
        {
            int source = j;
            int destination = i;
            if(i != j)
            {
                adjacency_matrix[source][destination] = 1;
            }
        }
    }
    printf("%.2f seconds to generate the graph.\n", omp_get_wtime() - start);
}

int main(int argc, char* argv[])
{
    // We do not need argc, this line silences potential compilation warnings.
    (void) argc;
    // We do not need argv, this line silences potential compilation warnings.
    (void) argv;

    printf("This program has two graph generators: generate_nice_graph and generate_sneaky_graph. If you intend to submit, your code will be timed on the sneaky graph, remember to try both.\n");

    // Get the time at the very start.
    double start = omp_get_wtime();
    
    generate_nice_graph();
 
    /// The array in which each vertex pagerank is stored.
    double pagerank[GRAPH_ORDER];
    calculate_pagerank(pagerank);
 
    // Calculates the sum of all pageranks. It should be 1.0, so it can be used as a quick verification.
    double sum_ranks = 0.0;
    for(int i = 0; i < GRAPH_ORDER; i++)
    {
        if(i % 100 == 0)
        {
            printf("PageRank of vertex %d: %.6f\n", i, pagerank[i]);
        }
        sum_ranks += pagerank[i];
    }
    printf("Sum of all pageranks = %.12f, total diff = %.12f, max diff = %.12f and min diff = %.12f.\n", sum_ranks, total_diff, max_diff, min_diff);
    double end = omp_get_wtime();
 
    printf("Total time taken: %.2f seconds.\n", end - start);
 
    return 0;
}
