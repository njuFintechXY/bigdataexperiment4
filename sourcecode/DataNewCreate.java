package skl;

import java.io.IOException;
import java.util.List;
import java.util.ArrayList;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
//import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.util.LineReader;

import org.apdplat.word.WordSegmenter;
import org.apdplat.word.segmentation.Word;

//用于将新闻标题转换为多维向量，预测集

public class DataNewCreate{
	//实现流程与训练集相似
	//为减小结果文件的大小，map输出key为股票代码+新闻标题，值为相应向量
	public static class DNCMapper extends Mapper<LongWritable,Text,Text,Text>{
		private String[] tez;//特征词
		private String pattern = "[^a-zA-Z0-9\u4e00-\u9fa5 -/\\.:]";//去除非法字符
		
		protected void setup(Context context) {
			tez = context.getConfiguration().getStrings("tez");
		}
		public void map(LongWritable offset,Text value,Context context) throws IOException,InterruptedException{
			int[] vec = new int[tez.length];
			String line = value.toString().replace("\t", "  ");//统一分隔符
			line = line.replaceAll(pattern, "");//去除错误数据，乱码
			String[] linesplit = line.split("  ");//fulldata文件，数据集有6列，标题为第5个
			
			if(linesplit.length != 6)return;
			List<Word> words = WordSegmenter.seg(linesplit[4]);
			for(Word w:words) {
				String wo = w.getText();
				for(int i=0;i<tez.length;i++) {
					if(tez[i].equals(wo)) {
						vec[i]+=1;
						break;
					}
				}
			}
			String valueres = Integer.toString(vec[0]);
			for(int j = 1; j<tez.length;j++) {
				valueres += " "+Integer.toString(vec[j]);
			}
			Text keyresult = new Text();
			Text valueresult = new Text();
			keyresult.set(linesplit[0]+linesplit[4]);
			valueresult.set(valueres);
			context.write(keyresult, valueresult);
		}
		
	}
	
	public static String[] gettez(Configuration conf, Path inputpath) throws IOException{
		//获取特征词
		ArrayList<String> tzc = new ArrayList<String>();
		Text line = new Text();
		FileSystem fs = FileSystem.get(conf);
		FSDataInputStream fsi = fs.open(inputpath);
		LineReader lr = new LineReader(fsi,conf);
		while(lr.readLine(line)>0) {
			tzc.add(line.toString());
		}
		lr.close();
		String[] b = (String[])tzc.toArray(new String[tzc.size()]);
		return b;
	}
	
	public static void main(String[] args) throws Exception{
		if(args.length != 3) {
			System.err.println("Usage: Datacreate <in> <ci.txt> <out>");
			System.exit(2);
		}
		Configuration conf = new Configuration();
		Path inputPath = new Path(args[0]);
		Path ciPath = new Path(args[1]);
		Path outputPath = new Path(args[2]);
		
		String[] tzc = gettez(conf,ciPath);
		conf.setStrings("tez", tzc);
		FileSystem fs = FileSystem.get(conf);
		if(fs.exists(outputPath))
			fs.delete(outputPath,true);
		Job job = Job.getInstance(conf,"dncjob");
		job.setJarByClass(DataNewCreate.class);
		job.setMapperClass(DNCMapper.class);
		job.setNumReduceTasks(1);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		//job.setOutputFormatClass(SequenceFileOutputFormat.class);
		FileInputFormat.addInputPath(job, inputPath);
		FileOutputFormat.setOutputPath(job, outputPath);
		System.exit(job.waitForCompletion(true)?0:1);
	}
}




