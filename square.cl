__kernel void square(

        __global int* input,

        __global int* output,

        const unsigned int count

)	
{
 
        const size_t id = get_global_id(0);
 
        if( id < count )
 
                output[id] = input[id]*input[id];
 
}
