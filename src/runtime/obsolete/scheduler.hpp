#ifndef SCHEDULER_HPP_
#define SCHEDULER_HPP_

//#include <mpi.h>
#include <iostream>
#include <string>
using namespace std;

namespace exatn {
namespace numerics {
	int proc_cnt; //currently serves as the resource count

	struct Task {
		string cmds;
		int num_procs;
	};

	struct Job {
		int id;
		Task tsk;
		string System;
		Job *next_job;

		Job (){ }

		Job(Task t, string s)
		{
			tsk=t;
			System=s;
			id=-1;
			next_job=NULL;
		}

		Job(Task t, string s, int i)
		{
			tsk=t;
			System=s;
			id=i;
			next_job=NULL;
		}
	};

	class Job_Queue {
	private: 
		int jobid_cnt;
		int qcnt;
	public:
		Job *first_job;
		Job *last_job;
		
		Job_Queue();

		//returns true if the queue is empty
		bool queue_empty();

		void print_queue();

		Job* get_next_job();

		Job* get_job_with_id(int id);

		void add_job(Job *j, bool is_newj, bool is_runj);

		void remove_job(Job *j, bool is_runj);
	};


	class Scheduler {
		public:
		Job_Queue *waitq;
		Job_Queue *runq; 

		Scheduler();

		void Job_Submit(string cmds, int np);

		int Launch_Job( );

		void Job_Done(int jb_id);
	};
}
}
#endif //SCHEDULER_HPP_
