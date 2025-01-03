// Code generated by cudago. Edit at your own risk.
package cudago

import (
    "github.com/InternatBlackhole/cudago/cuda"
	"unsafe"
)


//here just to force usage of unsafe package
var __urejanje_o_useless_var__ unsafe.Pointer = nil

const (
	KeyUrejanje_o = "urejanje_o"
)


type bitonicsortoArgs struct {
    a uintptr
    len int32
    k int32
    j int32

}

/*var (
    bitonicsortoArgs = bitonicsortoArgs{}

)*/







func BitonicSortO(grid, block cuda.Dim3, a uintptr, len int32, k int32, j int32) error {
	err := autoloadLib_urejanje_o()
	if err != nil {
		return err
	}
	kern, err := getKernel("urejanje_o", "bitonicSortO")
	if err != nil {
		return err
	}
	
	params := bitonicsortoArgs{
	    a: a,
	    len: len,
	    k: k,
	    j: j,
	
	}
	
	return kern.Launch(grid, block, unsafe.Pointer(&params.a), unsafe.Pointer(&params.len), unsafe.Pointer(&params.k), unsafe.Pointer(&params.j))
}

func BitonicSortOEx(grid, block cuda.Dim3, sharedMem uint64, stream *cuda.Stream, a uintptr, len int32, k int32, j int32) error {
	err := autoloadLib_urejanje_o()
	if err != nil {
		return err
	}
	kern, err := getKernel("urejanje_o", "bitonicSortO")
	if err != nil {
		return err
	}
	
	params := bitonicsortoArgs{
	    a: a,
	    len: len,
	    k: k,
	    j: j,
	
	}
	
	return kern.LaunchEx(grid, block, sharedMem, stream, unsafe.Pointer(&params.a), unsafe.Pointer(&params.len), unsafe.Pointer(&params.k), unsafe.Pointer(&params.j))
}



var loaded_urejanje_o = false


func autoloadLib_urejanje_o() error {
	if loaded_urejanje_o {
		return nil
	}
	err := InitLibrary([]byte(Urejanje_o_ptxCode), "urejanje_o")
	if err != nil {
		return err
	}
	loaded_urejanje_o = true
	return nil
}

const Urejanje_o_ptxCode = `//
// Generated by NVIDIA NVVM Compiler
//
// Compiler Build ID: CL-34431801
// Cuda compilation tools, release 12.6, V12.6.20
// Based on NVVM 7.0.1
//

.version 8.5
.target sm_52
.address_size 64

	// .globl	bitonicSortO

.visible .entry bitonicSortO(
	.param .u64 bitonicSortO_param_0,
	.param .u32 bitonicSortO_param_1,
	.param .u32 bitonicSortO_param_2,
	.param .u32 bitonicSortO_param_3
)
{
	.reg .pred 	%p<7>;
	.reg .b32 	%r<17>;
	.reg .b64 	%rd<7>;


	ld.param.u64 	%rd4, [bitonicSortO_param_0];
	ld.param.u32 	%r9, [bitonicSortO_param_1];
	ld.param.u32 	%r10, [bitonicSortO_param_2];
	ld.param.u32 	%r11, [bitonicSortO_param_3];
	mov.u32 	%r1, %ntid.x;
	mov.u32 	%r12, %ctaid.x;
	mov.u32 	%r13, %tid.x;
	mad.lo.s32 	%r16, %r12, %r1, %r13;
	setp.ge.s32 	%p1, %r16, %r9;
	@%p1 bra 	$L__BB0_8;

	cvta.to.global.u64 	%rd1, %rd4;
	mov.u32 	%r14, %nctaid.x;
	mul.lo.s32 	%r3, %r14, %r1;
	bra.uni 	$L__BB0_2;

$L__BB0_5:
	setp.le.s32 	%p5, %r6, %r7;
	@%p5 bra 	$L__BB0_7;
	bra.uni 	$L__BB0_6;

$L__BB0_2:
	xor.b32  	%r5, %r16, %r11;
	setp.le.s32 	%p2, %r5, %r16;
	@%p2 bra 	$L__BB0_7;

	and.b32  	%r15, %r16, %r10;
	setp.eq.s32 	%p3, %r15, 0;
	mul.wide.s32 	%rd5, %r16, 4;
	add.s64 	%rd2, %rd1, %rd5;
	ld.global.u32 	%r6, [%rd2];
	mul.wide.s32 	%rd6, %r5, 4;
	add.s64 	%rd3, %rd1, %rd6;
	ld.global.u32 	%r7, [%rd3];
	@%p3 bra 	$L__BB0_5;

	setp.lt.s32 	%p4, %r6, %r7;
	@%p4 bra 	$L__BB0_6;
	bra.uni 	$L__BB0_7;

$L__BB0_6:
	st.global.u32 	[%rd2], %r7;
	st.global.u32 	[%rd3], %r6;

$L__BB0_7:
	add.s32 	%r16, %r16, %r3;
	setp.lt.s32 	%p6, %r16, %r9;
	@%p6 bra 	$L__BB0_2;

$L__BB0_8:
	ret;

}

`
