#include "scheduler.hpp"

namespace exatn {
namespace numerics {

	Job_Queue::Job_Queue()
	{
		jobid_cnt=0;
		qcnt=0;
		first_job=NULL;
		last_job=NULL;
	}

	//returns true if the queue is empty
	bool Job_Queue::queue_empty()
	{
		return qcnt <= 0;
	}

	void Job_Queue::print_queue()
	{
		Job *tmp = first_job;
		while(tmp != NULL)
		{
			cout << tmp->id << endl << tmp->tsk.cmds << tmp->tsk.num_procs << endl << tmp->System << endl << endl;
			tmp=tmp->next_job;
		}
	}

	Job* Job_Queue::get_next_job() 
	{
		Job *temp=first_job;
		while(temp != NULL)
		{
			if(temp->tsk.num_procs <= proc_cnt) break;
			temp=temp->next_job;
		}
		
		if(temp == NULL) return NULL;

		return new Job(temp->tsk, temp->System, temp->id);
	}

	Job* Job_Queue::get_job_with_id(int id)
	{
		Job *temp=first_job;
		while(temp != NULL)
		{
			if(temp->id == id) break;
			temp=temp->next_job;
		}

		if(temp == NULL) return NULL;
		
		return new Job(temp->tsk, temp->System, temp->id);
	}

	void Job_Queue::add_job(Job *j, bool is_newj, bool is_runj)
	{
		//TODO: implement more complete error handling
		if(j == NULL) return;

		qcnt++;
		jobid_cnt++;
		if(is_newj) j->id=jobid_cnt;
		if(is_runj) proc_cnt-=j->tsk.num_procs;

		if(first_job==NULL)
		{
			first_job=j;
			last_job=j;
			j=NULL;
		}
		else
		{
			last_job->next_job=j;
			last_job=j;
			j=NULL;
		}
	}

	void Job_Queue::remove_job(Job *j, bool is_runj)
	{
		//TODO: implement more complete error handling
		if(j == NULL) return;

		Job *cur=new Job();
		Job *prev=new Job();
		cur=first_job;
		while(cur != NULL)
		{
			if(cur->id == j->id && cur->tsk.cmds == j->tsk.cmds && cur->System == j->System && cur->tsk.num_procs == j->tsk.num_procs)
			{
				if(cur==first_job)
					first_job = first_job->next_job;
				else
				{
					prev->next_job = cur->next_job;
					if(prev->next_job==NULL)
						last_job=prev;
				}
				if(is_runj) proc_cnt+=j->tsk.num_procs;

				delete cur;
				qcnt--;
				break;
			}
			else
			{
				prev = cur;
				cur=cur->next_job;
			}
		}
	}


	Scheduler::Scheduler () {
		waitq = new Job_Queue();
		runq = new Job_Queue(); 
	}

	void Scheduler::Job_Submit(string cmds, int np)
	{
		//create task
		Task tsk;
		tsk.cmds=cmds;
		tsk.num_procs=np;

		//create job
		Job *jb=new Job(tsk, "NaN");
		
		//add job to wait queue
		waitq->add_job(jb,true,false);
	}

	int Scheduler::Launch_Job( )
	{
		//get next job
		Job* jb = waitq->get_next_job();
		int id = jb->id;
	
		//add job to running queue
		runq->add_job(jb,false,true);

		//remove job from wait queue
		waitq->remove_job(jb,false);
		
		return id;
	}

	void Scheduler::Job_Done(int jb_id)
	{
		Job *jb = runq->get_job_with_id(jb_id);

		//remove job from run job queue
		runq->remove_job(jb,true);
	}

}
}

