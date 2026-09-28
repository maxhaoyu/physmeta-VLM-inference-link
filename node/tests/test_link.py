"""Real HTTP queue + agent tests; inference alone is replaced by a CPU fixture."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import urllib.error
import zipfile
from http.server import ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'node'))
import agent

class LinkTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        with patch.dict(os.environ, {'INFERENCE_DB':str(self.root/'queue.db'),
                'INFERENCE_STORAGE':str(self.root/'storage'), 'INFERENCE_NODE_TOKEN':'n'*64,
                'INFERENCE_UPLOAD_TOKEN':'u'*64}):
            spec=importlib.util.spec_from_file_location('queue_under_test', ROOT/'server/server.py')
            self.srv=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.srv)
        self.srv.init_db()
        self.http=ThreadingHTTPServer(('127.0.0.1',0),self.srv.Handler)
        self.http.daemon_threads=True
        self.thread=threading.Thread(target=self.http.serve_forever,daemon=True);self.thread.start()
        self.addCleanup(self.stop_server)
        self.config={'server':f'http://127.0.0.1:{self.http.server_port}', 'token':'n'*64,
            'node_id':'integration', 'jobs_root':self.root/'jobs', 'heartbeat_seconds':.1,
            'network_retry_seconds':2, 'retry_interval_seconds':.05}
        self.original_urlopen=agent._urlopen

    def stop_server(self):
        self.http.shutdown();self.http.server_close();self.thread.join()

    def job(self):
        jid=self.srv.new_job_id();data=b'synthetic drawing';p=self.root/(jid+'.png');p.write_bytes(data)
        c=self.srv.get_db();c.execute('INSERT INTO inference_jobs(id,input_path,input_sha256,input_bytes,filename,options) VALUES(?,?,?,?,?,?)',
            (jid,str(p),hashlib.sha256(data).hexdigest(),len(data),'synthetic.png','{}'));c.commit();c.close()
        result=agent.request_json(self.config,'POST','/api/inference/claim',{'wait_seconds':0})
        return result['job'],result['run_token']

    def artifact(self, name='result.zip', payload=None):
        p=self.root/name
        with zipfile.ZipFile(p,'w') as z:z.writestr('result.json',json.dumps(payload or {'boxes':[]}))
        return p

    def test_long_inference_renews_lease_while_another_node_polls(self):
        self.srv.CLAIM_TIMEOUT_SECONDS=1;self.srv.PROCESSING_TIMEOUT_SECONDS=1
        job,token=self.job()
        def slow(*_):
            deadline=time.monotonic()+3.2
            while time.monotonic()<deadline:
                other=agent.request_json({**self.config,'node_id':'second'},'POST','/api/inference/claim',{'wait_seconds':0})
                self.assertNotIn('job',other)
                time.sleep(.15)
            return {'boxes':[]}
        with patch.object(agent,'run_inference',side_effect=slow):agent.run_job(self.config,job,token)
        self.assertEqual(self.srv.job_row(job['id'])['status'],'done')
        self.assertFalse((self.config['jobs_root']/job['id']).exists())

    def test_network_reconnect_download_heartbeat_and_upload(self):
        job,token=self.job();failures={'/input':2,'/progress':2,'/result':2}
        def flaky(req,timeout):
            suffix=next((k for k in failures if req.full_url.endswith(k)),None)
            if suffix and failures[suffix]>0:
                failures[suffix]-=1;raise urllib.error.URLError('injected disconnection')
            return self.original_urlopen(req,timeout)
        with patch.object(agent,'_urlopen',side_effect=flaky),patch.object(agent,'run_inference',return_value={'boxes':[]}) as inference:
            agent.run_job(self.config,job,token)
        self.assertEqual(inference.call_count,1)
        self.assertEqual(failures,dict.fromkeys(failures,0))
        self.assertEqual(self.srv.job_row(job['id'])['status'],'done')

    def test_ack_lost_after_commit_retries_same_result_without_rerun(self):
        job,token=self.job();lost=False
        def lost_ack(req,timeout):
            nonlocal lost
            result=self.original_urlopen(req,timeout)
            if req.full_url.endswith('/result') and not lost:
                result.read();result.close();lost=True
                raise urllib.error.URLError('response lost after server committed')
            return result
        with patch.object(agent,'_urlopen',side_effect=lost_ack),patch.object(agent,'run_inference',return_value={'boxes':[{'final_text':'25'}]}) as inference:
            agent.run_job(self.config,job,token)
        self.assertEqual(inference.call_count,1)
        self.assertEqual(len(list((self.root/'storage/result').glob('*.zip'))),1)
        self.assertEqual(self.srv.job_row(job['id'])['status'],'done')

    def test_committed_result_rejects_different_retry(self):
        job,token=self.job();p=self.artifact()
        agent.upload_result(self.config,job['id'],token,p)
        p2=self.artifact('different.zip',{'boxes':[1]})
        with self.assertRaises(agent.AgentLeaseLost):agent.upload_result(self.config,job['id'],token,p2)
        stored=self.srv.job_row(job['id'])['result_path']
        self.assertEqual(Path(stored).read_bytes(),p.read_bytes())

    def test_reassigned_job_rejects_old_worker_and_keeps_local_result(self):
        job,token=self.job()
        def reassign(*_):
            c=self.srv.get_db();c.execute("UPDATE inference_jobs SET run_token='new-owner' WHERE id=?",(job['id'],));c.commit();c.close()
            return {'boxes':[]}
        with patch.object(agent,'run_inference',side_effect=reassign):
            with self.assertRaises(agent.AgentLeaseLost):agent.run_job(self.config,job,token)
        self.assertTrue((self.config['jobs_root']/job['id']/'result.json').is_file())
        self.assertEqual(self.srv.job_row(job['id'])['run_token'],'new-owner')
        self.assertNotEqual(self.srv.job_row(job['id'])['status'],'done')

    def test_missing_heartbeat_reclaims_crashed_worker(self):
        job,token=self.job();c=self.srv.get_db()
        c.execute("UPDATE inference_jobs SET status='processing', heartbeat_at=datetime('now','-7200 seconds') WHERE id=?",(job['id'],));c.commit();c.close()
        next_claim=agent.request_json(self.config,'POST','/api/inference/claim',{'wait_seconds':0})
        self.assertEqual(next_claim['job']['id'],job['id']);self.assertNotEqual(next_claim['run_token'],token)

    def test_retention_zero_preserves_old_results(self):
        job,token=self.job();agent.upload_result(self.config,job['id'],token,self.artifact())
        c=self.srv.get_db();c.execute("UPDATE inference_jobs SET updated_at=datetime('now','-90 days')");c.commit();c.close()
        self.srv.RETENTION_SECONDS=0
        self.assertEqual(self.srv._cleanup_expired(),0)
        self.assertTrue(Path(self.srv.job_row(job['id'])['result_path']).is_file())

if __name__=='__main__':unittest.main()
