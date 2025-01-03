// Code generated by cudago. Edit at your own risk.
package cudago

import (
    "github.com/InternatBlackhole/cudago/cuda"
	"unsafe"
)


//here just to force usage of unsafe package
var __razlika_4_useless_var__ unsafe.Pointer = nil

const (
	KeyRazlika_4 = "razlika_4"
)


type vectorsubtract4Args struct {
    c uintptr
    a uintptr
    b uintptr
    len int32

}

/*var (
    vectorsubtract4Args = vectorsubtract4Args{}

)*/







func VectorSubtract4(grid, block cuda.Dim3, c uintptr, a uintptr, b uintptr, len int32) error {
	err := autoloadLib_razlika_4()
	if err != nil {
		return err
	}
	kern, err := getKernel("razlika_4", "vectorSubtract4")
	if err != nil {
		return err
	}
	
	params := vectorsubtract4Args{
	    c: c,
	    a: a,
	    b: b,
	    len: len,
	
	}
	
	return kern.Launch(grid, block, unsafe.Pointer(&params.c), unsafe.Pointer(&params.a), unsafe.Pointer(&params.b), unsafe.Pointer(&params.len))
}

func VectorSubtract4Ex(grid, block cuda.Dim3, sharedMem uint64, stream *cuda.Stream, c uintptr, a uintptr, b uintptr, len int32) error {
	err := autoloadLib_razlika_4()
	if err != nil {
		return err
	}
	kern, err := getKernel("razlika_4", "vectorSubtract4")
	if err != nil {
		return err
	}
	
	params := vectorsubtract4Args{
	    c: c,
	    a: a,
	    b: b,
	    len: len,
	
	}
	
	return kern.LaunchEx(grid, block, sharedMem, stream, unsafe.Pointer(&params.c), unsafe.Pointer(&params.a), unsafe.Pointer(&params.b), unsafe.Pointer(&params.len))
}



var loaded_razlika_4 = false


func autoloadLib_razlika_4() error {
	if loaded_razlika_4 {
		return nil
	}
	err := InitLibrary([]byte(Razlika_4_ptxCode), "razlika_4")
	if err != nil {
		return err
	}
	loaded_razlika_4 = true
	return nil
}

const Razlika_4_ptxCode = `//
// Generated by NVIDIA NVVM Compiler
//
// Compiler Build ID: CL-34431801
// Cuda compilation tools, release 12.6, V12.6.20
// Based on NVVM 7.0.1
//

.version 8.5
.target sm_52
.address_size 64

	// .globl	vectorSubtract4

.visible .entry vectorSubtract4(
	.param .u64 vectorSubtract4_param_0,
	.param .u64 vectorSubtract4_param_1,
	.param .u64 vectorSubtract4_param_2,
	.param .u32 vectorSubtract4_param_3
)
{
	.reg .pred 	%p<3>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<11>;
	.reg .b64 	%rd<11>;


	ld.param.u64 	%rd4, [vectorSubtract4_param_0];
	ld.param.u64 	%rd5, [vectorSubtract4_param_1];
	ld.param.u64 	%rd6, [vectorSubtract4_param_2];
	ld.param.u32 	%r6, [vectorSubtract4_param_3];
	mov.u32 	%r1, %ntid.x;
	mov.u32 	%r7, %ctaid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r10, %r7, %r1, %r8;
	setp.ge.s32 	%p1, %r10, %r6;
	@%p1 bra 	$L__BB0_3;

	mov.u32 	%r9, %nctaid.x;
	mul.lo.s32 	%r3, %r9, %r1;
	cvta.to.global.u64 	%rd1, %rd5;
	cvta.to.global.u64 	%rd2, %rd6;
	cvta.to.global.u64 	%rd3, %rd4;

$L__BB0_2:
	mul.wide.s32 	%rd7, %r10, 4;
	add.s64 	%rd8, %rd1, %rd7;
	add.s64 	%rd9, %rd2, %rd7;
	ld.global.nc.f32 	%f1, [%rd9];
	ld.global.nc.f32 	%f2, [%rd8];
	sub.f32 	%f3, %f2, %f1;
	add.s64 	%rd10, %rd3, %rd7;
	st.global.f32 	[%rd10], %f3;
	add.s32 	%r10, %r10, %r3;
	setp.lt.s32 	%p2, %r10, %r6;
	@%p2 bra 	$L__BB0_2;

$L__BB0_3:
	ret;

}

`
