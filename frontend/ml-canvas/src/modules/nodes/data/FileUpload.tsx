import React, { useState, useCallback } from 'react';
import { Upload, FileSpreadsheet, AlertCircle, X } from 'lucide-react';
import axios from 'axios';

interface FileUploadProps {
  onUploadComplete: (datasetId: string, datasetName: string) => void;
  onCancel: () => void;
}

export const FileUpload: React.FC<FileUploadProps> = ({ onUploadComplete, onCancel }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setIsDragging(true);
    } else if (e.type === 'dragleave') {
      setIsDragging(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFiles(e.dataTransfer.files[0]);
    }
  }, []);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFiles(e.target.files[0]);
    }
  };

  const handleFiles = async (file: File) => {
    setUploading(true);
    setError(null);
    setProgress(0);

    const formData = new FormData();
    formData.append('file', file);

    try {
      // Use direct axios call to match backend route structure /data/api/upload
      const response = await axios.post('/data/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / (progressEvent.total || 1));
          setProgress(percentCompleted);
        },
      });

      if (response.data && response.data.success) {
        // Backend returns source_id
        const datasetId = response.data.source_id;
        const datasetName = response.data.file_info?.filename || file.name;
        onUploadComplete(datasetId, datasetName);
      } else {
        throw new Error(response.data?.message || 'Upload failed');
      }
    } catch (err: any) {
      console.error('Upload failed:', err);
      setError(err.response?.data?.detail || err.message || 'Failed to upload file. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="p-4 border rounded-lg bg-slate-50 dark:bg-slate-900 dark:border-slate-700 relative">
      <button 
        onClick={onCancel}
        className="absolute top-2 right-2 p-1 hover:bg-slate-200 dark:hover:bg-slate-800 rounded-full text-slate-500 dark:text-slate-400"
      >
        <X size={16} />
      </button>

      <h3 className="font-medium mb-4 flex items-center gap-2 text-slate-900 dark:text-slate-100">
        <Upload size={18} className="text-blue-600 dark:text-blue-400" />
        Upload New Dataset
      </h3>

      <div
        className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
          isDragging 
            ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20 dark:border-blue-400' 
            : 'border-slate-300 dark:border-slate-700 hover:border-slate-400 dark:hover:border-slate-600'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        {uploading ? (
          <div className="space-y-3">
            <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2.5">
              <div 
                className="bg-blue-600 h-2.5 rounded-full transition-all duration-300" 
                style={{ width: `${progress}%` }}
              ></div>
            </div>
            <p className="text-sm text-slate-600 dark:text-slate-300">Uploading... {progress}%</p>
          </div>
        ) : (
          <>
            <FileSpreadsheet className="w-10 h-10 text-slate-400 dark:text-slate-500 mx-auto mb-3" />
            <p className="text-sm text-slate-600 dark:text-slate-300 mb-2">
              Drag and drop your file here, or{' '}
              <label className="text-blue-600 dark:text-blue-400 hover:underline cursor-pointer font-medium">
                browse
                <input
                  type="file"
                  className="hidden"
                  accept=".csv,.xlsx,.parquet,.json"
                  onChange={handleChange}
                />
              </label>
            </p>
            <p className="text-xs text-slate-400">
              Supports CSV, Excel, Parquet, JSON
            </p>
          </>
        )}
      </div>

      {error && (
        <div className="mt-4 p-3 bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-400 text-sm rounded-md flex items-start gap-2">
          <AlertCircle size={16} className="mt-0.5 shrink-0" />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
};
