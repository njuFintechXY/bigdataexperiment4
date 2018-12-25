package skl;

import java.io.IOException;
import java.util.ArrayList;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.util.LineReader;

public class KNN{
	//map阶段计算出每条新闻最邻近的k个训练样本，输出<新闻，情感标签列表>
	//reduce阶段转化为情感输出<新闻，情感>
	
	public static int getdistance(String a,int[] b) {
		//计算两个点的距离
		//考虑到点集非常稀疏，这里直接取一阶距离（绝对值）
		String[] astring = a.split(" ");
		int dis = 0;
		if(astring.length!=b.length-1)
			return 0;
		else {
			for(int i = 0;i<astring.length;i++) {
				dis += Math.abs(Integer.parseInt(astring[i])-b[i]);
			}
			return dis;
		}
	}
	
	public static class Knnmap extends Mapper<Text,Text,Text,Text>{
		
		private int[][] yangben;
		private int k,m;
		
		protected void setup(Context context) {
			//获取训练集和k
			String[] yangbens= context.getConfiguration().getStrings("yangben");
			m = yangbens.length;
			int n = yangbens[0].split(" ").length+1;
			int[][] temp = new int[m][n];
			for(int i=0;i<m;i++) {
				String[] c = yangbens[i].split("\t");
				if(c.length!=2)continue;
				String[] s = c[0].split(" ");
				for(int j=0;j<n-1;j++) {
					temp[i][j]=Integer.parseInt(s[j]);
				}
				temp[i][n-1]=Integer.parseInt(c[1]);
			}
			yangben = temp;
			k = context.getConfiguration().getInt("k", 1);
		}
		public void map(Text key, Text value, Context context) throws IOException,InterruptedException{
			int[] distances = new int[k];
			int[] lables = new int[k];//情感标签，默认为0
			int i;
			for(i=0;i<k;i++) {
				distances[i]=Integer.MAX_VALUE;
			}
			int distance;
			
			for(int t=0;t<m;t++) {
				int[] s = yangben[t];
				distance = getdistance(value.toString(),s);
				for(i=0;i<k;i++) {
					if(distance<distances[i]) {
						distances[i]=distance;
						lables[i]=s[s.length-1];
						break;
					}
				}
			}
			String valueres = Integer.toString(lables[0]);
			for(int j = 1; j<k;j++) {
				valueres += " "+Integer.toString(lables[j]);
			}
			context.write(key, new Text(valueres));
			//context.write(key, value);
		}
	}
	
	public static class Knnreduce extends Reducer<Text,Text,Text,Text>{
		String[] qinggan = {"positive","negative","neutral"};
		
		public void reduce(Text key,Iterable<Text> values,Context context) throws IOException,InterruptedException{
			int predict=0;
			for(Text v:values) {
				String[] lables = v.toString().split(" ");
				int[] temp = new int[3];
				for(int i=0;i<lables.length;i++) {
					try {
						temp[Integer.parseInt(lables[i])-1]+=1;
					}catch(Exception e) {
						e.printStackTrace();
					}
				}
				for(int j = 1;j<3;j++) {
					if(temp[j]>temp[predict])
						predict=j;
				}
			}
			context.write(key, new Text(qinggan[predict]));
		}
	}
	
	public static String[] gettrain(Configuration conf, Path inputpath) throws IOException{
		//获取训练集
		ArrayList<String> train = new ArrayList<String>();
		Text line = new Text();
		FileSystem fs = FileSystem.get(conf);
		FSDataInputStream fsi = fs.open(inputpath);
		LineReader lr = new LineReader(fsi,conf);
		while(lr.readLine(line)>0) {
			train.add(line.toString());
		}
		lr.close();
		String[] b = (String[])train.toArray(new String[train.size()]);
		return b;
	}
	
	public static void main(String[] args) throws Exception{
		if(args.length != 3) {
			System.err.println("Usage: knn <train_in> <predict_in> <out>");
			System.exit(2);
		}
		Configuration conf = new Configuration();
		Path inputPath1 = new Path(args[0]);//训练集
		Path inputPath2 = new Path(args[1]);//预测集
		Path outputPath = new Path(args[2]);
		String[] yangben = gettrain(conf,inputPath1);
		conf.setStrings("yangben", yangben);
		//System.out.println(yangben.length);
		
		FileSystem fs = FileSystem.get(conf);
		if(fs.exists(outputPath))
			fs.delete(outputPath,true);
		Job job = Job.getInstance(conf,"knnjob");
		job.setJarByClass(KNN.class);
		job.setInputFormatClass(KeyValueTextInputFormat.class);
		job.setMapperClass(Knnmap.class);
		job.setReducerClass(Knnreduce.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		FileInputFormat.setInputPaths(job, inputPath2);
		FileOutputFormat.setOutputPath(job, outputPath);
		System.exit(job.waitForCompletion(true)?0:1);
	}
}




