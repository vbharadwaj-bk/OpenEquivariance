using namespace std;

template<int template_int>
__global__ void f3(int test) {  
    //*result = sizeof(T); 
    printf("Hello, my argument is %d\n", test);
    printf("Hello, my template parameter is %d\n", template_int);
}