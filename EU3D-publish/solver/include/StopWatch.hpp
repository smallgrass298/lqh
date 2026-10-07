#ifndef STOPWATCH
#define STOPWATCH
#include "TimeBase.hpp"
#include <iostream>
#include "Global.hpp"
template <typename T>
class basic_stopwatch;
typedef basic_stopwatch<TimeBase> Stopwatch;

template <typename T>
class basic_stopwatch
{
	T BaseTimer;

public:
	/* Create the stopwatch */
	explicit basic_stopwatch(bool start);
	explicit basic_stopwatch(char const *activity = "Stopwatch", bool start = true);
	/* basic_stopwatch(std::ostream& log, char const* activity = "Stopwatch", bool start = true); */

	/* Destroy the stopwatch */
	~basic_stopwatch();

	/* Get the last wathed time */
	unsigned LapGet() const;

	/* Judge if the stopwatch is running */
	bool IsStarted() const;

	/* Show the waited time, while keep running */
	void Show(char const *event_name = "show");

	/* Start the stopwatch */
	void Start(char const *event_name = "start");

	/* Stop the watch */
	void Stop(char const *event_name = "stop");

private:
};

template <typename T>
basic_stopwatch<T>::basic_stopwatch(bool start)
{
	if (start)
		Start("start");
}

template <typename T>
basic_stopwatch<T>::basic_stopwatch(char const *activity, bool start)
{
	std::cout << "Activity:\t" << activity << std::endl;
	if (start)
		Start();
}

template <typename T>
basic_stopwatch<T>::~basic_stopwatch()
{
	Stop();
}

template <typename T>
unsigned basic_stopwatch<T>::LapGet() const
{
	return BaseTimer.GetMs();
}

template <typename T>
bool basic_stopwatch<T>::IsStarted() const
{
	return BaseTimer.IsStart()();
}

template <typename T>
void basic_stopwatch<T>::Show(char const *event_name)
{
	std::cout << "Activity:\t" << event_name << "\t" << BaseTimer.GetMs() << " mS" << std::endl;
}

template <typename T>
void basic_stopwatch<T>::Start(char const *event_name)
{
	std::cout << "Process " << myid << ":\t" << event_name << std::endl;
	BaseTimer.Start();
}

template <typename T>
void basic_stopwatch<T>::Stop(char const *event_name)
{
	std::cout << "Process " << myid << ":\t" << event_name << "\t" << BaseTimer.GetMs() << " mS" << std::endl;
	BaseTimer.Clear();
}
#endif
