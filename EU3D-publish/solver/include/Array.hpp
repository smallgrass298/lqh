#pragma once
#include <assert.h>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <utility>
#include <tuple>
namespace ARRAY
{
	template <class Lam, size_t... Is>
	inline constexpr void Loop(Lam lam, std::index_sequence<Is...>)
	{
		(lam(Is), ...);
	};

	template <class T, size_t Dim>
	class Array
	{
	private:
		T *buf = nullptr;
		size_t Len = 0;
		size_t EachDim[Dim] = {};

		template <size_t... Is>
		inline const void PrintEachDim(std::index_sequence<Is...>)
		{
			auto PrintDim = [](auto i)
			{ std::cout << i << '\t'; };
			(PrintDim(EachDim[Is]), ...);
		}

		inline const size_t CalIndex(std::array<int, Dim> &CartIndex)
		{
			size_t FlatIndex = 0;
			Loop(
				// Omit the lambda always_inline attribute for Apple Clang compatibility.
				// Inlining is left to the compiler.
				[&](auto i) constexpr {
					FlatIndex = (CartIndex[i] + FlatIndex) * EachDim[i + 1];
				},
				std::make_index_sequence<Dim - 1>{});
			FlatIndex += CartIndex[Dim - 1];
			return FlatIndex;
		}

	public:
		Array() = default;

		/*constructor*/
		template <class... EACHDIM>
		Array(EACHDIM... each)
		{
			constexpr bool IsInteger =
				(true && ... && std::is_integral_v<decltype(each)>);
			/*ensure all input is integer*/
			static_assert(IsInteger == true);
			/*ensure the number of input parameters is the same as Dim*/
			static_assert(sizeof...(each) == Dim);

			/*get the length of all elements*/
			auto GetBufLen = [&]() -> size_t
			{ return (1 * ... * each); };
			Len = GetBufLen();
			/*Allocate buf*/
			buf = new T[Len];

			/*save the dim information to the array EachDim*/
			std::array<int, sizeof...(each)> EACH = {each...};
			Loop(
				// Omit the unsupported lambda attribute.
				[&](auto i) constexpr
				{ EachDim[i] = EACH[i]; },
				std::make_index_sequence<sizeof...(each)>{});
		}

		/*initial*/
		template <class... EACHDIM>
		void Initial(EACHDIM... each)
		{
			if (buf)
			{
				delete[] buf;
				buf = nullptr;
			}

			constexpr bool IsInteger =
				(true && ... && std::is_integral_v<decltype(each)>);
			/*ensure all input is integer*/
			static_assert(IsInteger == true);
			/*ensure the number of input parameters is the same as Dim*/
			static_assert(sizeof...(each) == Dim);

			/*get the length of all elements*/
			auto GetBufLen = [&]() -> size_t
			{ return (1 * ... * each); };
			Len = GetBufLen();
			/*Allocate buf*/
			buf = new T[Len];

			/*save the dim information to the array EachDim*/
			std::array<int, sizeof...(each)> EACH = {each...};
			Loop(
				// Omit the unsupported lambda attribute.
				[&](auto i) constexpr
				{ EachDim[i] = EACH[i]; },
				std::make_index_sequence<sizeof...(each)>{});
		}

		/*copy constructor*/
		template <class T_t, size_t Dim_t>
		Array(const Array<T_t, Dim_t> &array)
		{
			/*judge whether T_t can be convert into T*/
			constexpr auto IsConvert = std::is_convertible_v<T_t, T>;
			static_assert(IsConvert);
			/*Dim should be same*/
			static_assert(Dim == Dim_t);

			Len = array.GetLen();
			buf = new T[Len];
			auto array_ptr = array.Getbuf();
			for (int i = 0; i < Len; i++)
				buf[i] = array_ptr[i];
		}

		// /*move constructor*/
		// Array(Array &&array) : Len(array.Len)
		// {
		// 	buf = array.buf;
		// 	array.buf = nullptr;
		// }

		/*deconstructor*/
		~Array()
		{
			if (buf)
				delete[] buf;
			// std::cout<<"Object is deleted\n";
		}

		void Delete()
		{
			if (buf)
				delete[] buf;
			// std::cout << "Deleted\n";
		}

		/*print the dim infomation*/
		void GetDim()
		{
			std::cout << "Dim:" << Dim << "\nEach Dim:";
			PrintEachDim(std::make_index_sequence<Dim>{});
			std::cout << std::endl;
		}

		/*get Len*/
		size_t GetLen() const { return Len; }

		/*return buf*/
		T *Getbuf() const { return buf; }

		/*index the element*/
		template <class... INDEX>
		T &operator()(INDEX... index)
		{
			constexpr bool IsInteger =
				(true && ... && std::is_integral_v<decltype(index)>);
			/*ensure all input is integer*/
			static_assert(IsInteger == true);
			/*ensure the number of input is the same as Dim*/
			static_assert(sizeof...(index) == Dim);

			std::array<int, Dim> CartDim = {index...};
			auto Flat = CalIndex(CartDim);

			/*ensure the input index is within the data size*/
			// static_assert(Flat < Len);
#ifdef Debug
			auto IsInRange = true;
			Loop(
				[&](auto i)
				{
					IsInRange =
						IsInRange && CartDim[i] < EachDim[i] && CartDim[i] >= 0;
				},
				std::make_index_sequence<Dim>{});
			assert(IsInRange);
			assert(Flat >= 0 && Flat < Len);
#endif
			return buf[Flat];
		}

		/*index the element*/
		T &operator[](size_t Flat)
		{
#ifdef Debug
			assert(Flat >= 0 && Flat < Len);
#endif
			return buf[Flat];
		}

		/*= operator*/
		template <class T_t>
		Array<T, Dim> &operator=(const Array<T_t, Dim> &array)
		{
			// std::cout << "COPY\n";
			/*judge whether T_t can be convert into T*/
			constexpr auto IsConvert = std::is_convertible_v<T_t, T>;
			static_assert(IsConvert);

			if (buf)
				delete[] buf;
			Len = array.GetLen();
			buf = new T[Len];
			for (int i = 0; i < Dim; i++)
				EachDim[i] = array.EachDim[i];
			auto array_ptr = array.Getbuf();
			for (int i = 0; i < Len; i++)
				buf[i] = array_ptr[i];
			return *this;
		}

		/*= operator*/
		Array<T, Dim> &operator=(const Array<T, Dim> &array)
		{
			// std::cout << "COPY\n";
			if (this == &array)
				return *this;
			if (buf)
				delete[] buf;
			Len = array.GetLen();
			buf = new T[Len];
			for (int i = 0; i < Dim; i++)
				EachDim[i] = array.EachDim[i];
			auto array_ptr = array.Getbuf();
			for (int i = 0; i < Len; i++)
				buf[i] = array_ptr[i];
			return *this;
		}

		/*== operator*/
		template <class T_t, size_t Dim_t>
		bool operator==(Array<T_t, Dim_t> &array)
		{
			/*judge whether T_t can be convert into T*/
			constexpr auto IsSame = std::is_same_v<T, T_t>;
			static_assert(IsSame);
			/*Dim should be same*/
			static_assert(Dim == Dim_t);

			if (Len != array.GetLen())
				return false;
			else
			{
				auto result = true;
				auto array_ptr = array.Getbuf();
				for (int i = 0; i < Len; i++)
					result = result && (buf[i] == array_ptr[i]);
				return result;
			}
		}

		/*assign a specific value to the Array*/
		template <class Type>
		void Fill(Type value)
		{
			constexpr auto TypeOK = std::is_convertible_v<Type, T>;
			static_assert(TypeOK);
			for (int i = 0; i < Len; i++)
				buf[i] = value;
		}

		/* Get the Maximum value */
		inline const T MaxValue() const
		{
			T maxValue = buf[0];
			for (size_t i = 1; i < Len; i++)
			{
				if (buf[i] > maxValue)
				{
					maxValue = buf[i];
				}
			}
			return maxValue;
		}

		inline const size_t MaxPosition() const
		{
			T maxValue = buf[0];
			size_t maxPosition = 0;
			for (size_t i = 1; i < Len; i++)
			{
				if (buf[i] > maxValue)
				{
					maxValue = buf[i];
					maxPosition = i;
				}
			}
			return maxPosition;
		}

		/* Get the Minimum value */
		inline const T MinValue() const
		{
			T minValue = buf[0];
			for (size_t i = 1; i < Len; i++)
			{
				if (buf[i] < minValue)
				{
					minValue = buf[i];
				}
			}
			return minValue;
		}

		/* Get the Average value */
		inline const T AveValue() const
		{
			return Sum() / Len;
		}

		/* Summation function of the array */
		inline const T Sum() const
		{
			T sum = 0;
			for (size_t i = 0; i < Len; i++)
			{
				sum += buf[i];
			}
			return sum;
		}

		inline const T SumNoBoundary(int bc) const
		{
			T sum = 0;
			for (size_t i = 0; i < Len; i++)
			{
				size_t z = i % EachDim[2];
				size_t y = (i / EachDim[2]) % EachDim[1];
				size_t x = i / (EachDim[2] * EachDim[1]);
				if (x >= bc && y >= bc && z >= bc &&
					x < EachDim[0] - bc &&
					y < EachDim[1] - bc &&
					z < EachDim[2] - bc)
					sum += buf[i];
			}
			return sum;
		}

		/* Get the sum value of elements>0 */
		inline const T SumPositive() const
		{
			T sump = 0;
			for (size_t i = 0; i < Len; i++)
			{
				if (buf[i] > 0)
				{
					sump += buf[i];
				}
			}
			return sump;
		}

		/* Judge if data is nan */
		inline const bool IsNan() const
		{
			for (size_t i = 0; i < Len; i++)
			{
				if (std::isnan(buf[i]))
				{
#ifdef Debug
					std::cout << "Exist NAN\n";
#endif
					return true;
				}
			}
#ifdef Debug
			std::cout << "No NAN\n";
#endif
			return false;
		}

		/* Get information */
		inline void Print()
		{
			for (size_t i = 0; i < Len; i++)
				std::cout << buf[i] << '\t';
			std::cout << '\n';
		}
		inline size_t GetSize() const { return Len; }
		inline size_t GetDim() const { return Dim; }
		inline size_t GetNi() const { return EachDim[0]; }
		inline size_t GetNj() const { return EachDim[1]; }
		inline size_t GetNk() const { return EachDim[2]; }
		std::tuple<size_t, size_t, size_t> Get3DIndices(size_t m)
		{
			size_t k = m % EachDim[2];
			size_t j = (m / EachDim[2]) % EachDim[1];
			size_t i = m / (EachDim[2] * EachDim[1]);
			return std::make_tuple(i, j, k);
		}
	};
} // namespace ZCCTools
