#ifndef TIMEBASE
#define TIMEBASE
#include <chrono>
using namespace std::chrono;

class TimeBase
{
private:
	system_clock::time_point m_start;

public:
	TimeBase() : m_start(system_clock::time_point::min()) { ; }

	/* Clean the stopwatch */
	void Clear()
	{
		m_start = system_clock::time_point::min();
	}

	/* Whether the stopwatch is running */
	bool IsStart() const
	{
		return (m_start.time_since_epoch() != system_clock::duration(0));
	}

	/* start */
	void Start()
	{
		m_start = system_clock::now();
	}

	/* Get the time */
	unsigned long GetMs()
	{
		if (IsStart())
		{
			system_clock::duration diff;
			diff = system_clock::now() - m_start;
			return (unsigned)(duration_cast<milliseconds>(diff).count());
		}
		return 0;
	}
};
#endif