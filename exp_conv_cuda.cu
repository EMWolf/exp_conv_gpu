//

//  test_cuda.c

//

//

//  Created by Andrew Christlieb on 3/3/14.

//  This code is a CUDA kernal for domain decompositon

//  of the implcit maxwell solver.

//  The code is mearly a test code and use #define

//  where the varables should be done with ArgV ArgC.

//

//  This is mealy a 1D test code!  NOT GREAT, but menat

//  for learning!

//


//GPU Version: parallel add of arbitrary length.

#include <stdio.h>

#include <stdbool.h>

#include <math.h>

#include <cuda.h>


const int N = (int)pow(2,27);    /* num grid point */

const int M = 16;      /* max number mesh cells */



//each tid - denotes a subdominant up to M cells

//

//   |----------|----------|----------|...----------|-----| (note: last cell<M)

//     domain 0   domain 1   domain 2     domain K-2  domain K-1

//


/* 
Function name: debug_tool_output_to_file

Function type: host function

Purpose: Given a pointer to an array of single-digit integers of length N, outputs the array to a text file.
			The text file is displayed in rows of M entries.

Inputs: int * debugArray_ptr - a pointer to an array of single digit integers
			const int N - the length of the array
			const int M - the length of a row in the text file

Outputs: generates a file named debug_output.txt
*/


void debug_tool_output_to_file(int * debugArray_ptr, const int N, const int M){
    int k = N/M;           /* number of whole sub domains */

    int k_end = N%M;         /* number of points in last domain */

    int k_tot;             /* total number of sub domains */

    int loop_over_y_cells; /* number of mesh point in domain */

    int cell_index,i,j;     /* cells_index is the flattend index of the cell */

                           /* j is merely a counter */

    bool test;             /* flag to indicate if we have a single nonuniform

                               sub domain */

	char output;

    /* set up logic for when number of subdomains

     does not evenly divide total number of cells in a line. */

    if(k_end>0)

    {

        k_tot = k+1;       /* number of sub domains */

        test = true;       /* flag for indicating a single special domain */

    }

    else

    {

        k_tot = k;         /* number of sub domains */

        test = false;

    }
	
	
	FILE * fp;
	
	fp = fopen("debug_output.txt","w");
	
	for(i = 0; i< k_tot; i++){
		if((i==(k_tot-1))&&test){
			loop_over_y_cells = k_end;
		}
		else{
			loop_over_y_cells = M;
		}
		for(j = 0; j< loop_over_y_cells; j++ ){
			
			cell_index = i*M+j;
			output = (char)(((int)'0')+debugArray_ptr[cell_index]);
			fputc((int)output,fp);
		}
		fprintf(fp,"\n");
	}
	
	fclose(fp);
	
}


/*
Function name: localWeightAndSweep_L

Function type: CUDA kernel

Purpose: Performs the local deposit step then the local sweep step over each subdomain for the left integral part.

Inputs: float * JL_d - pointer to array on device storing the local integral values of the left part
		float * val_d - pointer to array on device storing the integrand values
		const int N - number of grid points in the array
		const int M - number of grid points in one subdomain
		float nu - numerical parameter equal to dx*alpha
		int * debugArray_d - pointer to array of single-digit integers on device; used for diagnostic purposes only

Outputs: JL_d is written to at all grid points
*/
__global__ void localWeightAndSweep_L(float *JL_d, float *val_d, const int N, const int M, float nu, int *debugArray_d)

{

    int k = N/M;           /* number of whole sub domains */

    int k_end = N%M;         /* number of points in last domain */

    int k_tot;             /* total number of sub domains */

    int loop_over_y_cells; /* number of mesh point in domain */

    int cell_index,j;     /* cells_index is the flattend index of the cell */

                           /* j is merely a counter */

    bool test;             /* flag to indicate if we have a single nonuniform

                               sub domain */
		

	int startLoopIndex; /* Loop start index in subdomain - should be 1 in first subdomain, 0 in others. */
	int endLoopIndex; /* Loop end index - should be k_end-1 in the last subdomain, M in others. */ 
		


	float ex = exp(-nu);


		/* Quadrature weights */
	float P = 1.0 - (1.0-ex)/nu;
	float Q = -ex+(1.0-ex)/nu;
	float R = (1.0-ex)/(nu*nu)-(1.0+ex)/(2.0*nu);



    /* set up logic for when number of subdomains

     does not evenly divide total number of cells in a line. */

    if(k_end>0)

    {

        k_tot = k+1;       /* number of sub domains */

        test = true;       /* flag for indicating a single special domain */

    }

    else

    {

        k_tot = k;         /* number of sub domains */

        test = false;

    }



    /* use unique tid to identify which sub domain */

    int tid = threadIdx.x+blockIdx.x*blockDim.x;  //index of sub domain



    /* let the kernel address more than a single sub domain */

    while(tid<k_tot){
		
		/* If we are in the first subdomain, the first point needs special treatment, as the usual quadrature stencil will extend outside of the domain. */
		/* We have that JL[0]=0. */
		if(tid==0)
		{
			startLoopIndex = 1;
			JL_d[0]=0.0;
		}
		else
		
			startLoopIndex = 0;

		
        if((tid==k_tot-1)&&(test))

            loop_over_y_cells= k_end;

        else

            loop_over_y_cells= M;

		
		/* If we are in the last subdomain, the last point needs special treatment, as the usual quadrature stencil will extend outside of the domain. */
		/* We have that JL[N-1] is NOT zero, so we need special treatment at the right endpoint for JL. */
		if (tid==k_tot-1)
		{
			endLoopIndex = loop_over_y_cells - 1;
			JL_d[N-1] = P*val_d[N-1]+Q*val_d[N-2]+R*(val_d[1]-2.0*val_d[N-1]+val_d[N-2]); /* For periodic integral */
		}
		else
			endLoopIndex = loop_over_y_cells;
		
		
		
		/* Perform the local weight step across the subdomain. 
		Every point in every subdomain (except for the domain boundary points) should be set here. */
        for(j=startLoopIndex;j<endLoopIndex; j++)

        {

            cell_index=j+tid*M;        /* Compute cell offset for cell j of */

                                       /* of sub domain domain tid */



                                       /* Compute integral */
			
			JL_d[cell_index]=P*val_d[cell_index]+Q*val_d[cell_index-1]+R*(val_d[cell_index+1]-2.0*val_d[cell_index]+val_d[cell_index-1]);
			//debugArray_d[cell_index] += 1;
        }
		
		
		/* Perform the local recursive sweep across the subdomain. */
        for(j=1; j<loop_over_y_cells; j++)

        {

            cell_index=j+tid*M;        /* Compute cell offset for cell j of */

                                       /* of sub domain domain tid */
			
			JL_d[cell_index]+=ex*JL_d[cell_index-1]; /* Perform recursive push */
			//debugArray_d[cell_index] += 1;

        }

        tid+=gridDim.x*blockDim.x;     /* jump to next sub domain this kernel */

                                       /* which is a whole cuda GRID away */

                                       /* grid dimension - gridDim.x*blockDim.x */

    };

}

/*
Function name: coarseSweep_L

Function type: CUDA kernel, but a serial process; should be called with one block/one thread

Purpose: performs the (serial) coarse grid sweep of the left integral part

Inputs: float * IL_d - pointer to array on device string of total integral value of the left part
		float * JL_d - pointer to array on device storing the local integral values of the left part
		const int N - number of grid points in the array
		const int M - number of grid points in one subdomain
		float nu - numerical parameter equal to dx*alpha
		int * debugArray_d - pointer to array of single-digit integers on device; used for diagnostic purposes only

Outputs: IL_d is written to on the coarse grid points, which for the left integral part are the last points of each subdomain, 
		with index of the form ((tid+1)*M-1). 
		At these points, these are the final integral values and need not be updated again.
*/

__global__ void coarseSweep_L(float * IL_d, float * JL_d, const int N, const int M, float nu, int *debugArray_d)
{

    int k = N/M;           /* number of whole sub domains */

    int k_end = N%M;         /* number of points in last domain */

    int k_tot;             /* total number of sub domains */


    int cell_index;     /* cells_index is the flattend index of the cell */

                           /* j is merely a counter */

    //bool test;             /* flag to indicate if we have a single nonuniform

                               //sub domain */
		


	float ex_subdom = exp(-nu*M);
	float ex_end = exp(-nu*k_end);


	float recursion_coeff;
	int subdom_offset;

    /* set up logic for when number of subdomains

     does not evenly divide total number of cells in a line. */

    if(k_end>0)

    {

        k_tot = k+1;       /* number of sub domains */

        //test = true;       /* flag for indicating a single special domain */

    }

    else

    {

        k_tot = k;         /* number of sub domains */

        //test = false;

    }



    /* use unique tid to identify which sub domain */

    int tid = threadIdx.x+blockIdx.x*blockDim.x;  //index of sub domain



    /* let the kernel address more than a single sub domain */

    while(tid<(k_tot-2)){

		if(tid==0){
			IL_d[M-1]=JL_d[M-1];
			//debugArray_d[M-1] += 1;
		}
		
		recursion_coeff = ex_subdom;
		subdom_offset = M;

        cell_index=M*(tid+1)-1;        /* Compute cell offset for cell j of */

                                       /* of sub domain domain tid */


			
		IL_d[cell_index+subdom_offset]=JL_d[cell_index+subdom_offset]+ IL_d[cell_index]*recursion_coeff;
		//debugArray_d[cell_index+subdom_offset] += 1;

        
		
        

        tid+=gridDim.x*blockDim.x;     /* jump to next sub domain this kernel */

                                       /* which is a whole cuda GRID away */

                                       /* grid dimension - gridDim.x*blockDim.x */

    };

}

/*
Function name: coarseToFineSweep_L

Function type: CUDA kernel

Purpose: performs the final coarse grid to fine grid sweep of the left integral part

Inputs: float * IL_d - pointer to array on device string of total integral value of the left part
		float * JL_d - pointer to array on device storing the local integral values of the left part
		const int N - number of grid points in the array
		const int M - number of grid points in one subdomain
		float nu - numerical parameter equal to dx*alpha
		int * debugArray_d - pointer to array of single-digit integers on device; used for diagnostic purposes only

Outputs: IL_d is written to everywhere in the domain, except for the endpoints which were already written to in coarseSweep_L
*/

__global__ void coarseToFineSweep_L(float * IL_d, float * JL_d, const int N, const int M, float nu, int * debugArray_d)
{

    int k = N/M;           /* number of whole sub domains */

    int k_end = N%M;         /* number of points in last domain */

    int k_tot;             /* total number of sub domains */

    int loop_over_y_cells; /* number of mesh point in domain */

    int cell_index,j;     /* cells_index is the flattend index of the cell */

                           /* j is merely a counter */

    bool test;             /* flag to indicate if we have a single nonuniform

                               sub domain */
		
	int startLoopIndex; /* Loop start index in subdomain. */
	int endLoopIndex; /* Loop end index. */
	int source_index;
	
	float ex = exp(-nu);



	float push_tracker;

    /* set up logic for when number of subdomains

     does not evenly divide total number of cells in a line. */

    if(k_end>0)

    {

        k_tot = k+1;       /* number of sub domains */

        test = true;       /* flag for indicating a single special domain */

    }

    else

    {

        k_tot = k;         /* number of sub domains */

        test = false;

    }



    /* use unique tid to identify which sub domain */

    int tid = threadIdx.x+blockIdx.x*blockDim.x;  //index of sub domain



    /* let the kernel address more than a single sub domain */

    while(tid<k_tot){
        if((tid==k_tot-1)&&(test))
		{
            loop_over_y_cells= k_end;

		}

        else
		{	
            loop_over_y_cells= M;
		
		}

		
        if(tid==k_tot-1)
		{
			source_index = M*(tid)-1;
			push_tracker = IL_d[source_index];
			startLoopIndex = 1;
			endLoopIndex = loop_over_y_cells+1;

		}
		else if(tid==0)
		{
			source_index = 0;
			push_tracker = (float)0;
			startLoopIndex = 1;
			endLoopIndex = loop_over_y_cells-1;
		}

        else
		{	
			source_index = M*(tid)-1;
			push_tracker = IL_d[source_index];
			startLoopIndex = 1;
			endLoopIndex = loop_over_y_cells;
		
		}



		
		/* Index of IL to be used as source for sweep - last grid point in subdom tid */
		//source_index = M*(tid+1)-1; 
		
		//push_tracker = IL_d[source_index];
		
        for(j=startLoopIndex;j<endLoopIndex; j++)

        {

            cell_index=source_index+j;        /* Compute cell offset for cell j of */

                                       /* of sub domain domain tid */

			
			IL_d[cell_index]= JL_d[cell_index]+ex*push_tracker;
			push_tracker = push_tracker*ex;
			
			//debugArray_d[cell_index] += 1;
        }

        tid+=gridDim.x*blockDim.x;     /* jump to next sub domain this kernel */

                                       /* which is a whole cuda GRID away */

                                       /* grid dimension - gridDim.x*blockDim.x */

    };

}


/*
Function name: localWeightAndSweep_R

Function type: CUDA kernel

Purpose: Performs the local deposit step then the local sweep step over each subdomain for the right integral part.

Inputs: float * JR_d - pointer to array on device storing the local integral values of the right part
		float * val_d - pointer to array on device storing the integrand values
		const int N - number of grid points in the array
		const int M - number of grid points in one subdomain
		float nu - numerical parameter equal to dx*alpha
		int * debugArray_d - pointer to array of single-digit integers on device; used for diagnostic purposes only

Outputs: JR_d is written to at all grid points
*/
__global__ void localWeightAndSweep_R(float * JR_d, float * val_d, const int N, const int M, float nu, int * debugArray_d)

{

    int k = N/M;           /* number of whole sub domains */

    int k_end = N%M;         /* number of points in last domain */

    int k_tot;             /* total number of sub domains */

    int loop_over_y_cells; /* number of mesh point in domain */

    int cell_index,j;     /* cells_index is the flattend index of the cell */

                           /* j is merely a counter */

    bool test;             /* flag to indicate if we have a single nonuniform

                               sub domain */
	
	int startLoopIndex; /* Loop start index in subdomain - should be 1 in first subdomain, 0 in others. */
	int endLoopIndex; /* Loop end index - should be k_end-1 in the last subdomain, M in others. */ 
		

	float ex = exp(-nu);

		/* Quadrature weights */
	float P = 1.0 - (1.0-ex)/nu;
	float Q = -ex+(1.0-ex)/nu;
	float R = (1.0-ex)/(nu*nu)-(1.0+ex)/(2.0*nu);



    /* set up logic for when number of subdomains

     does not evenly divide total number of cells in a line. */

    if(k_end>0)

    {

        k_tot = k+1;       /* number of sub domains */

        test = true;       /* flag for indicating a single special domain */

    }

    else

    {

        k_tot = k;         /* number of sub domains */

        test = false;

    }



    /* use unique tid to identify which sub domain */

    int tid = threadIdx.x+blockIdx.x*blockDim.x;  //index of sub domain



    /* let the krenal address more than a single sub domain */

    while(tid<k_tot){
		
        if((tid==k_tot-1)&&(test))

            loop_over_y_cells= k_end;

        else

            loop_over_y_cells= M;
		
		/* If we are in the first subdomain, the first point needs special treatment, as the usual quadrature stencil will extend outside of the domain. */
		/* We have that JR[0] is NOT zero, so we need special treatment of the left endpoint for JR. */
		if(tid==0)
		{
			startLoopIndex = 1;
			JR_d[0] = P*val_d[0]+Q*val_d[1]+R*(val_d[1]-2.0*val_d[0]+val_d[N-2]); /* For periodic integral */
		}
		else
		
			startLoopIndex = 0;

		
        

		
		/* If we are in the last subdomain, the last point needs special treatment, as the usual quadrature stencil will extend outside of the domain. */
		/* We have that JR[N-1]=0. */
		if (tid==k_tot-1)
		{
			endLoopIndex = loop_over_y_cells-1;
			JR_d[N-1]=0.0;
		}	
		else
			
			endLoopIndex = loop_over_y_cells;
		
		
		
		/* Perform the local weight step across the subdomain. */
        for(j=startLoopIndex;j<endLoopIndex; j++)

        {

            cell_index=tid*M+j;        /* Compute cell offset for cell j of */

                                       /* of sub domain domain tid */



                                       /* Compute integral */
			
			JR_d[cell_index]=P*val_d[cell_index]+Q*val_d[cell_index+1]+R*(val_d[cell_index+1]-2.0*val_d[cell_index]+val_d[cell_index-1]);
			//debugArray_d[cell_index] += 1;
        }
		
		
		/* Perform the local recursive sweep across the subdomain. */
        for(j=2;j<loop_over_y_cells+1; j++)

        {

            cell_index=tid*M+loop_over_y_cells-j;        /* Compute cell offset for cell j of */

                                       /* of sub domain domain tid */
			
			JR_d[cell_index]+=ex*JR_d[cell_index+1]; /* Perform recursive push */
			//debugArray_d[cell_index] += 1;

        }

        tid+=gridDim.x*blockDim.x;     /* jump to next sub domain this kernel */

                                       /* which is a whole cuda GRID away */

                                       /* grid dimension - gridDim.x*blockDim.x */

    };

}


/*
Function name: coarseSweep_R

Function type: CUDA kernel, but a serial process; should be called with one block/one thread

Purpose: performs the (serial) coarse grid sweep of the right integral part

Inputs: float * IR_d - pointer to array on device string of total integral value of the right part
		float * JR_d - pointer to array on device storing the local integral values of the right part
		const int N - number of grid points in the array
		const int M - number of grid points in one subdomain
		float nu - numerical parameter equal to dx*alpha
		int * debugArray_d - pointer to array of single-digit integers on device; used for diagnostic purposes only

Outputs: IR_d is written to on the coarse grid points, which for the right integral part are the first points of each subdomain, 
		with index of the form tid*M. 
		At these points, these are the final integral values and need not be updated again.
*/
__global__ void coarseSweep_R(float * IR_d, float * JR_d, const int N, const int M, float nu, int * debugArray_d)

{

    int k = N/M;           /* number of whole sub domains */

    int k_end = N%M;         /* number of points in last domain */

    int k_tot;             /* total number of sub domains */


    int cell_index;     /* cells_index is the flattend index of the cell */

                           /* j is merely a counter */

    bool test;             /* flag to indicate if we have a single nonuniform

                               sub domain */
	
	
	float ex_subdom = exp(-nu*M);
	float ex_end = exp(-nu*k_end);


	float recursion_coeff;
	int subdom_offset;

    /* set up logic for when number of subdomains

     does not evenly divide total number of cells in a line. */

    if(k_end>0)

    {

        k_tot = k+1;       /* number of sub domains */

        test = true;       /* flag for indicating a single special domain */

    }

    else

    {

        k_tot = k;         /* number of sub domains */

        test = false;

    }



    /* use unique tid to identify which sub domain */

    int tid = threadIdx.x+blockIdx.x*blockDim.x;  //index of sub domain



    /* let the kernel address more than a single sub domain */

    while(tid<k_tot-1){
		
		if(tid==0){
			if(test){
				recursion_coeff = ex_subdom;
				subdom_offset = M;
				cell_index = N-k_end;
			}
			else{
				recursion_coeff = ex_subdom;
				subdom_offset = M;
				cell_index = N-M;
			}
			IR_d[cell_index]=JR_d[cell_index];
			
			debugArray_d[cell_index] += 1;
		}
		else{
			if(test){
            	recursion_coeff = ex_subdom;
				subdom_offset = M;
				cell_index = N-k_end-tid*M;
			}	
			else{
				recursion_coeff = ex_subdom;
				subdom_offset = M;
				cell_index = N-(tid+1)*M;
			}
			//debugArray_d[cell_index] += 1;
		}

		

			
		IR_d[cell_index-subdom_offset]=JR_d[cell_index-subdom_offset] + IR_d[cell_index]*recursion_coeff;
		debugArray_d[cell_index-subdom_offset] += 1;

        
		
        

        tid+=gridDim.x*blockDim.x;     /* jump to next sub domain this kernel */

                                       /* which is a whole cuda GRID away */

                                       /* grid dimension - gridDim.x*blockDim.x */

    };


}


/*
Function name: coarseToFineSweep_R

Function type: CUDA kernel

Purpose: performs the final coarse grid to fine grid sweep of the right integral part

Inputs: float * IR_d - pointer to array on device string of total integral value of the right part
		float * JR_d - pointer to array on device storing the local integral values of the right part
		const int N - number of grid points in the array
		const int M - number of grid points in one subdomain
		float nu - numerical parameter equal to dx*alpha
		int * debugArray_d - pointer to array of single-digit integers on device; used for diagnostic purposes only

Outputs: IR_d is written to everywhere in the domain, except for the endpoints which were already written to in coarseSweep_R
*/
__global__ void coarseToFineSweep_R(float * IR_d, float * JR_d, const int N, const int M, float nu, int * debugArray_d)

{

    int k = N/M;           /* number of whole sub domains */

    int k_end = N%M;         /* number of points in last domain */

    int k_tot;             /* total number of sub domains */

    int loop_over_y_cells; /* number of mesh point in domain */

    int cell_index,j;     /* cells_index is the flattend index of the cell */

                           /* j is merely a counter */

    bool test;             /* flag to indicate if we have a single nonuniform

                               sub domain */
	
	int startLoopIndex; /* Loop start index in subdomain - should be 1 in first subdomain, 0 in others. */
	int endLoopIndex; /* Loop end index - should be k_end-1 in the last subdomain, M in others. */ 
	int source_index;


	float ex = exp(-nu);



	float push_tracker;

    /* set up logic for when number of subdomains

     does not evenly divide total number of cells in a line. */

    if(k_end>0)

    {

        k_tot = k+1;       /* number of sub domains */

        test = true;       /* flag for indicating a single special domain */

    }

    else

    {

        k_tot = k;         /* number of sub domains */

        test = false;

    }



    /* use unique tid to identify which sub domain */

    int tid = threadIdx.x+blockIdx.x*blockDim.x;  //index of sub domain



    /* let the kernel address more than a single sub domain */

    while(tid<k_tot){
        if((tid==k_tot-1)&&(test))
		{
            loop_over_y_cells= k_end;

		}

        else
		{	
            loop_over_y_cells= M;
		
		}

		
        if(tid==k_tot-1)
		{
			source_index = N-1;
			push_tracker = (float)0;
			startLoopIndex = 1;
			endLoopIndex = loop_over_y_cells-1;

		}

        else
		{	
			source_index = M*(tid+1);
			push_tracker = IR_d[source_index];
			startLoopIndex = 1;
			endLoopIndex = loop_over_y_cells;
		
		}
		
		
		
		
		
        for(j=startLoopIndex;j<endLoopIndex; j++)

        {

            cell_index=source_index-j;        /* Compute cell offset for cell j of */

                                       /* of sub domain domain tid */

			
			IR_d[cell_index]= JR_d[cell_index]+ex*push_tracker;
			push_tracker = push_tracker*ex;
			
			debugArray_d[cell_index] += 1;
        }

        tid+=gridDim.x*blockDim.x;     /* jump to next sub domain this kernel */

                                       /* which is a whole cuda GRID away */

                                       /* grid dimension - gridDim.x*blockDim.x */

    };

}

/*
Function name: vectorAdd

Function type: CUDA kernel

Purpose: adds two arrays on the device - I_d = IL_d + IR_d

Inputs: float * I_d - pointer to array on device containing the summed values
		float * IL_d - pointer to array on device
		float * IR_d - pointer to array on device
		const int N - number of grid points in the array
		const int M - number of grid points in one subdomain
		int * debugArray_d - pointer to array of single-digit integers on device; used for diagnostic purposes only
		
Outputs: I_d is written to at every grid point

*/

__global__ void vectorAdd(float * I_d, float * IL_d, float * IR_d, const int N, const int M, int * debugArray_d){

    int k = N/M;           /* number of whole sub domains */

    int k_end = N%M;         /* number of points in last domain */

    int k_tot;             /* total number of sub domains */

    int loop_over_y_cells; /* number of mesh point in domain */

    int cell_index,j;     /* cells_index is the flattend index of the cell */

                           /* j is merely a counter */

    bool test;             /* flag to indicate if we have a single nonuniform

                               sub domain */
	
	int startLoopIndex; /* Loop start index in subdomain - should be 1 in first subdomain, 0 in others. */
	int endLoopIndex; /* Loop end index - should be k_end-1 in the last subdomain, M in others. */ 



    /* set up logic for when number of subdomains

     does not evenly divide total number of cells in a line. */

    if(k_end>0)

    {

        k_tot = k+1;       /* number of sub domains */

        test = true;       /* flag for indicating a single special domain */

    }

    else

    {

        k_tot = k;         /* number of sub domians */

        test = false;

    }



    /* use uniuck tid to identify which sub domain */

    int tid = threadIdx.x+blockIdx.x*blockDim.x;  //index of sub domain



    /* let the krenal address more than a single sub domain */

    while(tid<k_tot){
		
		/* If we are in the first subdomain, the first point needs special treatment, as the usual quadrature stencil will extend outside of the domain. */
		/* We have that JL[0]=0. */

		
        if((tid==k_tot-1)&&(test))

            loop_over_y_cells= k_end;

        else

            loop_over_y_cells= M;


		startLoopIndex = 0;
		endLoopIndex = loop_over_y_cells;
		
		/* Perform the local weight step across the subdomain. */
        for(j=startLoopIndex;j<endLoopIndex; j++)

        {

            cell_index=j+tid*M;        /* Compute cell offset for cell j of */

                                       /* of sub domain domain tid */



                                       /* Compute integral */
			
			I_d[cell_index]=IL_d[cell_index]+IR_d[cell_index];
			
			
			//debugArray_d[cell_index] += 1;

        }
		
		
        tid+=gridDim.x*blockDim.x;     /* jump to next sub domain this kernel */

                                       /* which is a whole cuda GRID away */

                                       /* grid dimension - gridDim.x*blockDim.x */
    };

}



int main(void){

	clock_t start = clock(), setupTime, kernelTime, cleanupTime, testTime, diff;
	
	float L = 1.0;
	float dx = L/((float)(N-1));
	float nu = 1.0;
	float alpha = nu/dx;
	float x;
	float err = 0.0;
	float err_temp;
	
	int Nt = 1;
	
    float val[N];
	float I[N];    //host memory

	int debugArray[N];
	//int * debugArray_ptr = &debugArray;
	
	int * debugArray_d;

    float * val_d; // Integrand

	float * JL_d;
	float * JR_d;
	float * IL_d;
	float * IR_d;
	float * I_d;

    int num_B=1024,num_T=32;
	
	printf("Number of grid points: %i\n",N);
	
	printf("Number of time steps: %i\n",Nt);

    //allocate device memory

    cudaMalloc((void **) &val_d,sizeof(float)*N);

	cudaMalloc((void **) &JL_d,sizeof(float)*N);
	cudaMalloc((void **) &JR_d,sizeof(float)*N);
	cudaMalloc((void **) &IL_d,sizeof(float)*N);
	cudaMalloc((void **) &IR_d,sizeof(float)*N);
	cudaMalloc((void **) &I_d,sizeof(float)*N);

	cudaMalloc((void **) &debugArray_d,sizeof(int)*N);
    //Set Inital Condtion...

    for(int i=0;i<N;i++){

        val[i]=(float) 1;

		I[i]=(float) 0;
		
		debugArray[i] = 0;
    };



    //mem copy


    cudaMemcpy(val_d,val,sizeof(float)*N,cudaMemcpyHostToDevice);
	cudaMemcpy(JL_d,I,sizeof(float)*N,cudaMemcpyHostToDevice);
	cudaMemcpy(JR_d,I,sizeof(float)*N,cudaMemcpyHostToDevice);
	cudaMemcpy(IL_d,I,sizeof(float)*N,cudaMemcpyHostToDevice);
	cudaMemcpy(IR_d,I,sizeof(float)*N,cudaMemcpyHostToDevice);
	cudaMemcpy(I_d,I,sizeof(float)*N,cudaMemcpyHostToDevice);
	
	cudaMemcpy(debugArray_d,debugArray,sizeof(int)*N,cudaMemcpyHostToDevice);

	setupTime = clock();
	diff = setupTime-start;
	int sec = diff/ CLOCKS_PER_SEC;
	printf("Setup time: %d seconds\n", sec);
	
    //Call kernel
	for(int n = 0; n<Nt; n++){
    localWeightAndSweep_L<<<num_B,num_T>>>(JL_d,val_d,N,M,nu,debugArray_d);
	coarseSweep_L<<<1,1>>>(IL_d,JL_d,N,M,nu,debugArray_d);
	coarseToFineSweep_L<<<num_B,num_T>>>(IL_d,JL_d,N,M,nu,debugArray_d);
    localWeightAndSweep_R<<<num_B,num_T>>>(JR_d,val_d,N,M,nu,debugArray_d);
	coarseSweep_R<<<1,1>>>(IR_d,JR_d,N,M,nu,debugArray_d);
	coarseToFineSweep_R<<<num_B,num_T>>>(IR_d,JR_d,N,M,nu,debugArray_d);
	vectorAdd<<<num_B,num_T>>>(I_d,IL_d,IR_d,N,M,debugArray_d);
	}
	
	kernelTime = clock();
	diff = kernelTime-setupTime;
	sec = diff/ CLOCKS_PER_SEC;
	printf("Kernel time: %d seconds\n", sec);

    //mem copy

    cudaMemcpy(I,I_d,sizeof(float)*N,cudaMemcpyDeviceToHost);
	cudaMemcpy(debugArray,debugArray_d,sizeof(float)*N,cudaMemcpyDeviceToHost);

    cudaFree(val_d);
	cudaFree(JL_d);
	cudaFree(JR_d);
	cudaFree(IL_d);
	cudaFree(IR_d);
	cudaFree(I_d);
	cudaFree(debugArray_d);
	
	cleanupTime = clock();
	diff = cleanupTime-kernelTime;
	sec = diff/ CLOCKS_PER_SEC;
	printf("Clean up time: %d seconds\n",sec);
	
	/* Compute test integral value. */
	
	for(int j=0; j<N; j++){
		x = (float)j*dx;
		err_temp = abs(I[j]-(2.0-exp(-alpha*x)-exp(-alpha*(L-x))));
		
		if (err_temp>1.0e-6)
			printf("Error of %f at grid point j = %i \n", err_temp,j);
		
		if (err_temp>err){
			err = err_temp;
		};
		
	}
	
	printf("Maximum error: %f \n", err);
	
	testTime = clock();
	diff = testTime-cleanupTime;
	sec = diff/ CLOCKS_PER_SEC;
	printf("Test time: %d seconds\n",sec);
	

	//debug_tool_output_to_file( debugArray, N, M);
	
	return 0;


};