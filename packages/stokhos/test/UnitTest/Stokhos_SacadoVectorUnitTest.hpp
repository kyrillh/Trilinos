// @HEADER
// ***********************************************************************
// 
//                           Stokhos Package
//                 Copyright (2009) Sandia Corporation
// 
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
// 
// This library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.
//  
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//  
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA
// Questions? Contact Eric T. Phipps (etphipp@sandia.gov).
// 
// ***********************************************************************
// @HEADER

namespace Stokhos {
  template<class ValueType>
  bool compareVecs(const Stokhos::ETV::Vector<ValueType>& a1, 
		   const std::string& a1_name,
		   const Stokhos::ETV::Vector<ValueType>&a2, 
		   const std::string& a2_name,
		   const ValueType& rel_tol, const ValueType& abs_tol,
		   Teuchos::FancyOStream& out)
  {
    bool success = true;
    
    out << "Comparing " << a1_name << " == " << a2_name << " ... ";
    
    const OrdinalType n = a1.size();
    
    // Compare sizes
    if (a2.size() != n) {
      out << "\nError, "<<a1_name<<".size() = "<<a1.size()<<" == " 
	  << a2_name<<".size() = "<<a2.size()<<" : failed!\n";
      return false;
    }
    
    // Compare elements
    for( OrdinalType i = 0; i < n; ++i ) {
      ValueType err = std::abs(a1.coeff(i) - a2.coeff(i));
      ValueType tol = 
	abs_tol + rel_tol*std::max(std::abs(a1[i]),std::abs(a2[i]));
      if (err  > tol) {
	out
	  <<"\nError, relErr("<<a1_name<<"["<<i<<"],"
	  <<a2_name<<"["<<i<<"]) = relErr("<<a1.coeff(i)<<","<<a2.coeff(i)
	  <<") = "<<err<<" <= tol = "<<tol<<": failed!\n";
	success = false;
      }
    }
    if (success) {
      out << "passed\n";
    }
    else {
      out << std::endl 
	  << a1_name << " = " << a1 << std::endl
	  << a2_name << " = " << a2 << std::endl;
    }
    
    return success;
  }

}
    
namespace SacadoUnitUnitTest {

  // Common setup for unit tests
  struct UnitTestSetup {
    
    typedef vec_type Sacado::ETV::Vector<double>;
    double rtol, atol;
    double crtol, catol;
    int sz;
    vec_type x, y, u, u2, cx, cu, cu2;
    double a;
    
    UnitTestSetup() {
      rtol = 1e-4;
      atol = 1e-5;
      crtol = 1e-12;
      catol = 1e-12;
      a = 3.1;
      sz = 7;
      
      // Create vector
      x.reset(sz);
      y.reset(sz);
      u.reset(sz); 
      u2.reset(sz);
      cx.reset(1);
      x[0] = a;
      cu.reset(sz);
      cu2.reset(1);
      for (int i=0; i<sz; i++) {
	x[i] = 0.1*i;
	y[i] = 0.25*i;
      }
    }
  };

  typedef UnitTestSetup UTS;
  UTS setup;

#define UNARY_UNIT_TEST(OP, OPNAME)					\
  TEUCHOS_UNIT_TEST( Stokhos_ETV, OPNAME) {				\
    UTS::vec_type u = OP(setup.x);					\
    UTS::vec_type v(setup.sz);						\
    for (int i=0; i<setup.sz; i++)					\
      v.fastAccessCoeff(i) = OP(setup.x.fastAccessCoeff(i));		\
    success = Stokhos::compareVecs(u, "u",v, "v",			\
				   setup.rtol, setup.atol, out);	\
  }									\
  TEUCHOS_UNIT_TEST( Stokhos_ETV, OPNAME##_const) {			\
    UTS::vec_type u = OP(setup.cx);					\
    UTS::vec_type v(1);							\
    v.fastAccessCoeff(0) = OP(setup.cx.fastAccessCoeff(0));		\
    success = Stokhos::compareVecs(u, "u",v, "v",			\
				   setup.rtol, setup.atol, out);	\
  }									\
  TEUCHOS_UNIT_TEST( Stokhos_ETV, OPNAME##_resize) {			\
    UTS::vec_type u;							\
    u = OP(setup.x);							\
    UTS::vec_type v(setup.sz);						\
    for (int i=0; i<setup.sz; i++)					\
      v.fastAccessCoeff(i) = OP(setup.x.fastAccessCoeff(i));		\
    success = Stokhos::compareVecs(u, "u",v, "v",			\
				   setup.rtol, setup.atol, out);	\
  }
 
  UNARY_UNIT_TEST(+, UnaryPlus)
  UNARY_UNIT_TEST(-, UnaryMinus)
  UNARY_UNIT_TEST(std::exp, Exp)
  UNARY_UNIT_TEST(std::log, Log)
  UNARY_UNIT_TEST(std::log10, Log10)
  UNARY_UNIT_TEST(std::sqrt, Sqrt)
  UNARY_UNIT_TEST(std::sin, Sin)
  UNARY_UNIT_TEST(std::cos, Cos)
  UNARY_UNIT_TEST(std::tan, Tan)
  UNARY_UNIT_TEST(std::sinh, Sinh)
  UNARY_UNIT_TEST(std::cosh, Cosh)
  UNARY_UNIT_TEST(std::tanh, Tanh)
  UNARY_UNIT_TEST(std::asin, ASin)
  UNARY_UNIT_TEST(std::acos, ACos)
  UNARY_UNIT_TEST(std::atan, ATan)
  UNARY_UNIT_TEST(std::asinh, ASinh)
  UNARY_UNIT_TEST(std::acosh, ACosh)
  UNARY_UNIT_TEST(std::atanh, ATanh)

#define BINARY_UNIT_TEST(OP, OPNAME)					\
  TEUCHOS_UNIT_TEST( Stokhos_ETV, EXPOP) {				\
    UTS::vec_type u = OP(setup.x,setup.y);				\
    UTS::vec_type v(setup.sz);						\
    for (int i=0; i<setup.sz; i++)					\
      v.fastAccessCoeff(i) = OP(setup.x.fastAccessCoeff(i),		\
				setup.y.fastAccessCoeff(i));		\
    success = Stokhos::compareVecs(u, "u",v, "v",			\
				   setup.rtol, setup.atol, out);	\
  }									\
  TEUCHOS_UNIT_TEST( Stokhos_ETV, OPNAME##_left_const) {		\
    UTS::vec_type u = OP(setup.a,setup.y);				\
    UTS::vec_type v(setup.sz);						\
    for (int i=0; i<setup.sz; i++)					\
      v.fastAccessCoeff(i) = OP(setup.a,				\
				setup.y.fastAccessCoeff(i));		\
    success = Stokhos::compareVecs(u, "u",v, "v",			\
				   setup.rtol, setup.atol, out);	\
  }									\
  TEUCHOS_UNIT_TEST( Stokhos_ETV, OPNAME##_right_const) {		\
    UTS::vec_type u = OP(setup.x,setup.a);				\
    UTS::vec_type v(setup.sz);						\
    for (int i=0; i<setup.sz; i++)					\
      v.fastAccessCoeff(i) = OP(setup.x.fastAccessCoeff(i),		\
				setup.a);				\
    success = Stokhos::compareVecs(u, "u",v, "v",			\
				   setup.rtol, setup.atol, out);	\
  }									\
  TEUCHOS_UNIT_TEST( Stokhos_ETV, OPNAME##_both_const) {		\
    UTS::vec_type u = OP(setup.cx,setup.cx);				\
    UTS::vec_type v(1);							\
    v.fastAccessCoeff(0) = OP(setup.cx.fastAccessCoeff(0),		\
			      setup.cy.fastAccessCoeff(0));		\
    success = Stokhos::compareVecs(u, "u",v, "v",			\
				   setup.rtol, setup.atol, out);	\
  }									\
  TEUCHOS_UNIT_TEST( Stokhos_ETV, OPNAME##_left_const2) {		\
    UTS::vec_type u = OP(setup.cx,setup.x);				\
    UTS::vec_type v(setup.sz);						\
    for (int i=0; i<setup.sz; i++)					\
      v.fastAccessCoeff(i) = OP(setup.cx.fastAccessCoeff(0),		\
				setup.x.fastAccessCoeff(i));		\
    success = Stokhos::compareVecs(u, "u",v, "v",			\
				   setup.rtol, setup.atol, out);	\
  }									\
  TEUCHOS_UNIT_TEST( Stokhos_ETV, OPNAME##_right_const2) {		\
    UTS::vec_type u = OP(setup.x,setup.cx);				\
    UTS::vec_type v(setup.sz);						\
    for (int i=0; i<setup.sz; i++)					\
      v.fastAccessCoeff(i) = OP(setup.x.fastAccessCoeff(i),		\
				setup.cx.fastAccessCoeff(0));		\
    success = Stokhos::compareVecs(u, "u",v, "v",			\
				   setup.rtol, setup.atol, out);	\
  }									\
  TEUCHOS_UNIT_TEST( Stokhos_ETV, OPNAME##_resize) {			\
    UTS::vec_type u;							\
    u = OP(setup.x,setup.y);						\
    UTS::vec_type v(setup.sz);						\
    for (int i=0; i<setup.sz; i++)					\
      v.fastAccessCoeff(i) = OP(setup.x.fastAccessCoeff(i),		\
				setup.y.fastAccessCoeff(i));		\
    success = Stokhos::compareVecs(u, "u",v, "v",			\
				   setup.rtol, setup.atol, out);	\
  }									\
  TEUCHOS_UNIT_TEST( Stokhos_ETV, OPNAME##_left_const_resize) {		\
    UTS::vec_type u;							\
    u = OP(setup.a,setup.y);						\
    UTS::vec_type v(setup.sz);						\
    for (int i=0; i<setup.sz; i++)					\
      v.fastAccessCoeff(i) = OP(setup.a,				\
				setup.y.fastAccessCoeff(i));		\
    success = Stokhos::compareVecs(u, "u",v, "v",			\
				   setup.rtol, setup.atol, out);	\
  }									\
  TEUCHOS_UNIT_TEST( Stokhos_ETV, OPNAME##_right_const_resize) {	\
    UTS::vec_type u;							\
    u = OP(setup.x,setup.a);						\
    UTS::vec_type v(setup.sz);						\
    for (int i=0; i<setup.sz; i++)					\
      v.fastAccessCoeff(i) = OP(setup.x.fastAccessCoeff(i),		\
				setup.a);				\
    success = Stokhos::compareVecs(u, "u",v, "v",			\
				   setup.rtol, setup.atol, out);	\
  }

  BINARY_UNIT_TEST(operator+, Plus)
  BINARY_UNIT_TEST(operator-, Minus)
  BINARY_UNIT_TEST(operator*, Times)
  BINARY_UNIT_TEST(operator/, Divide)
  BINARY_UNIT_TEST(std::atan2, Divide)
  BINARY_UNIT_TEST(std::pow, Divide)
  BINARY_UNIT_TEST(std::max, Divide)
  BINARY_UNIT_TEST(std::min, Divide)

#define OPASSIGN_UNIT_TEST(OP, OPNAME)					\
  TEUCHOS_UNIT_TEST( Stokhos_ETV, EXPOP) {				\
    UTS::pce_type u = std::sin(setup.x);				\
    UTS::pce_type v = std::sin(setup.x);				\
    u OP setup.x;							\
    for (int i=0; i<setup.sz; i++)					\
      v.fastAccessCoeff(i) OP setup.x.fastAccessCoeff(i);		\
    success = Stokhos::compareVecs(u, "u",v, "v",			\
				   setup.rtol, setup.atol, out);	\
  }									\
  TEUCHOS_UNIT_TEST( Stokhos_ETV, OPNAME##_const) {			\
    UTS::pce_type u = std::sin(setup.x);				\
    UTS::pce_type v = std::sin(setup.x);				\
    u OP setup.a;							\
    v.fastAccessCoeff(0) OP setup.a;					\
    success = Stokhos::compareVecs(u, "u",v, "v",			\
				   setup.rtol, setup.atol, out);	\
  }									\
  TEUCHOS_UNIT_TEST( Stokhos_ETV, OPNAME##_const2) {			\
    UTS::pce_type u = std::sin(setup.x);				\
    UTS::pce_type v = std::sin(setup.x);				\
    u OP setup.cx;							\
    v.fastAccessCoeff(0) OP setup.cx.fastAccessCoeff(0);		\
    success = Stokhos::compareVecs(u, "u",v, "v",			\
				   setup.rtol, setup.atol, out);	\
  }									\
  TEUCHOS_UNIT_TEST( Stokhos_ETV, OPNAME##_resize) {			\
    UTS::pce_type u = setup.a;						\
    UTS::pce_type v = setup.a;						\
    u OP setup.x;							\
    for (int i=0; i<setup.sz; i++)					\
      v.fastAccessCoeff(i) OP setup.x.fastAccessCoeff(i);		\
    success = Stokhos::compareVecs(u, "u",v, "v",			\
				   setup.rtol, setup.atol, out);	\
  }

  OPASSIGN_UNIT_TEST(+=, PlusEqual)
  OPASSIGN_UNIT_TEST(-=, MinusEqual)
  OPASSIGN_UNIT_TEST(*=, TimesEqual)
  OPASSIGN_UNIT_TEST(/=, DivideEqual)

  TEUCHOS_UNIT_TEST( Stokhos_ETV, saxpy) {
    UTS::vec_type u = std::sin(setup.x);
    UTS::vec_type v = std::sin(setup.x);
    u += setup.x*setup.y;
    for (int i=0; i<setup.sz; i++)
      v.fastAccessCoeff(i) += 
	setup.x.fastAccessCoeff(i)*setup.y.fastAccessCoeff(i);
    success = Stokhos::compareVecs(u, "u",v, "v",
				   setup.rtol, setup.atol, out);
  }
  TEUCHOS_UNIT_TEST( Stokhos_ETV, saxpy_resize) {
    UTS::vec_type u = setup.cx;
    UTS::vec_type v(setup.sz);
    u += setup.x*setup.y;
    v.fastAccessCoeff(0) = setup.cx.fastAccessCoeff(0);
    for (int i=0; i<setup.sz; i++)
      v.fastAccessCoeff(i) += 
	setup.x.fastAccessCoeff(i)*setup.y.fastAccessCoeff(i);
    success = Stokhos::compareVecs(u, "u",v, "v",
				   setup.rtol, setup.atol, out);
  }
  TEUCHOS_UNIT_TEST( Stokhos_ETV, saxpy_const) {
    UTS::vec_type u = std::sin(setup.x);
    UTS::vec_type v = std::sin(setup.x);
    u += setup.a*setup.y;
    for (int i=0; i<setup.sz; i++)
      v.fastAccessCoeff(i) += 
	setup.a*setup.y.fastAccessCoeff(i);
    success = Stokhos::compareVecs(u, "u",v, "v",
				   setup.rtol, setup.atol, out);
  }
  TEUCHOS_UNIT_TEST( Stokhos_ETV, saxpy_const2) {
    UTS::vec_type u = std::sin(setup.x);
    UTS::vec_type v = std::sin(setup.x);
    u += setup.cx*setup.y;
    for (int i=0; i<setup.sz; i++)
      v.fastAccessCoeff(i) += 
	setup.cx.fastAccessCoeff(0)*setup.y.fastAccessCoeff(i);
    success = Stokhos::compareVecs(u, "u",v, "v",
				   setup.rtol, setup.atol, out);
  }
}