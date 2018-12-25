package skl;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.util.LineReader;

public class NaiveBayes{
	/*
	 * 先运行一个训练mapreduce生成训练结果
	 * 分类mapreduce将该结果读成全局文件分配给各节点，其输入文件为预测集
	 */
	
	
	public static class TrainMapper extends Mapper<Text, Text, Text, IntWritable>{
		private final static IntWritable one = new IntWritable(1);
		private Text word;
	
		public void map(Text key, Text value, Context context)throws IOException, InterruptedException {
			String vals[], temp;
			int i;
			word = new Text();
			context.write(value, one);//Fyi
			vals = key.toString().split(" ");
		
			for(i=0;i<vals.length;i++) {
				temp = value.toString()+"#"+Integer.toString(i)+"#"+vals[i];
				word.set(temp);
				context.write(word, one);//用于计算FxYij,格式为Yi#属性#属性值
			}
		}
	}
	
	public static class TrainReducer extends Reducer<Text,IntWritable,Text,IntWritable>{
		private IntWritable result = new IntWritable();
		public void reduce(Text key,Iterable<IntWritable> values,Context context) throws IOException,InterruptedException{
			int sum = 0;
			for(IntWritable val:values) {
				sum += val.get();
			}
			result.set(sum);
			context.write(key, result);
		}
	}
	
	public static class TestMapper extends Mapper<Text,Text,Text,Text>{
		private Map<String,Double> FY = new HashMap<String,Double>();//分类Yi出现的频度
		private Map<String,Double> FXY = new HashMap<String,Double>();//每个属性值xij在Yi中出现的频度
		String[] qinggan = {"positive","negative","neutral"};
		
		public void setup(Context context) {
			//获取训练生成的数据
			String[] FYstring = context.getConfiguration().getStrings("FY");
			String[] FXYstring = context.getConfiguration().getStrings("FXY");
			for(int i=0;i<FYstring.length;i++) {
				String[] strsplit = FYstring[i].split("\t");
				if(strsplit.length != 2)continue;
				FY.put(strsplit[0], Double.parseDouble(strsplit[1])/1000.0);
			}
			for(int j=0;j<FXYstring.length;j++) {
				String[] strsplit = FXYstring[j].split("\t");
				if(strsplit.length != 2)continue;
				FXY.put(strsplit[0], Double.parseDouble(strsplit[1])/1000.0);
			}
		}
		
		public void map(Text key,Text value,Context context) throws IOException,InterruptedException{
			String[] vec = value.toString().split(" ");
			double maxF = Double.MIN_VALUE;
			int index = -1;
			for(String s:FY.keySet()){
				double FXYi = 0.0;
				double Yi = FY.get(s);
				for(int i=0;i<vec.length;i++) {
					double FxYij;
					try {
						FxYij = FXY.get(s+"#"+Integer.toString(i)+"#"+vec[i]);
					}catch(Exception e) {
						FxYij = 0;
					}
					FXYi = FXYi + FxYij;//考虑到2000维的数据会导致乘积近似为0的问题，取加法运算
				}
				//context.write(key, new Text(s+"#"+Double.toString(Yi)+"#"+Double.toString(FXYi)));
				if(Yi*FXYi>maxF) {
					maxF=Yi*FXYi;
					index = Integer.parseInt(s)-1;
				}
			}
			if(index<qinggan.length&&index>-1)context.write(key, new Text(qinggan[index]));
			//else context.write(key, new Text(Integer.toString(index)+"#"+Double.toString(FXY.get("1#0#0"))));
		}
	}
	
	public static Map<String,String[]> gettrainF(Configuration conf, Path outputpath) throws IOException{
		//根据训练输出结果来生成频率文件
		ArrayList<String> trainFY = new ArrayList<String>();//Yi
		ArrayList<String> trainFXY = new ArrayList<String>();//Yi+#+...
		Text line = new Text();
		FileSystem fs = FileSystem.get(conf);
		FileStatus[] outputfiles = fs.listStatus(outputpath);
		FSDataInputStream fsi = null;
		for(int i=0;i<outputfiles.length;i++) {
			fsi = fs.open(outputfiles[i].getPath());
			LineReader lr = new LineReader(fsi,conf);
			while(lr.readLine(line)>0) {
				if(line.toString().split("#").length>1)
					trainFXY.add(line.toString());
				else
					trainFY.add(line.toString());
			}
			lr.close();
		}
		Map<String,String[]> result = new HashMap<String,String[]>();
		result.put("FY", (String[])trainFY.toArray(new String[trainFY.size()]));
		result.put("FXY", (String[])trainFXY.toArray(new String[trainFXY.size()]));
		return result;
	}
	
	public static void main(String[] args) throws Exception{
		if(args.length != 3) {
			System.err.println("Usage: naivebayes <train_in> <predict_in> <out>");
			System.exit(2);
		}
		Configuration conf = new Configuration();
		Path traininput = new Path(args[0]);//训练集
		Path predictinput = new Path(args[1]);//预测集
		Path tempPath = new Path(args[0]+"_temp");//训练产生的中间频度结果
		Path outputPath = new Path(args[2]);
		
		
		
		FileSystem fs = FileSystem.get(conf);

		if(fs.exists(tempPath))
			fs.delete(tempPath,true);
		Job trainjob = Job.getInstance(conf, "trainjob");
		trainjob.setJarByClass(NaiveBayes.class);
		trainjob.setMapperClass(TrainMapper.class);
		trainjob.setCombinerClass(TrainReducer.class);
		trainjob.setReducerClass(TrainReducer.class);
		trainjob.setOutputKeyClass(Text.class);
		trainjob.setOutputValueClass(IntWritable.class);
		trainjob.setInputFormatClass(KeyValueTextInputFormat.class);
		FileInputFormat.addInputPath(trainjob, traininput);
		FileOutputFormat.setOutputPath(trainjob, tempPath);
		if(trainjob.waitForCompletion(true)) {
			//获取训练结果，传入全局文件
			Map<String,String[]> trainmapresult = gettrainF(conf,tempPath);
			conf.setStrings("FY", trainmapresult.get("FY"));
			conf.setStrings("FXY", trainmapresult.get("FXY"));
			//执行预测mapreduce
			if(fs.exists(outputPath))
				fs.delete(outputPath,true);
			Job testjob = Job.getInstance(conf,"testjob");
			testjob.setJarByClass(NaiveBayes.class);
			testjob.setMapperClass(TestMapper.class);
			testjob.setNumReduceTasks(1);
			testjob.setInputFormatClass(KeyValueTextInputFormat.class);
			testjob.setOutputKeyClass(Text.class);
			testjob.setOutputValueClass(Text.class);
			FileInputFormat.addInputPath(testjob, predictinput);
			FileOutputFormat.setOutputPath(testjob, outputPath);
			System.exit(testjob.waitForCompletion(true)?0:1);
		}
	}
}